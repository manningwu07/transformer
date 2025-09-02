package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"gonum.org/v1/gonum/mat"
)

// How many times does attn --> mlp happen

// Adam Optimizer global vars
var (
	embM, embV *mat.Dense
	embT       int
)

type TrainingConfig struct {
	// Core transformer parameters
	DModel     int // model width
	HiddenSize int // MLP hidden
	VocabSize  int // |V|
	NumHeads   int // attention heads
	SeqLen     int // max prefix length (context length)
	AttnLR     float64
	MLPLR      float64
	UnembedLR  float64

	// Optimization/training wheel parameters
	NormLR      float64
	WarmupSteps int     // linear warmup steps
	DecaySteps  int     // cosine decay steps after warmup (0 = none)
	AdamBeta1   float64 // default 0.9
	AdamBeta2   float64 // default 0.999
	AdamEps     float64 // default 1e-8

	MaxEpochs int     // maximum number of epochs
	Patience  int     // early stopping patience
	Epsilon   float64 // stop if loss < epsilon
	BatchSize int     // mini-batch size
	ValFrac   float64 // fraction of data held out for validation

	// Stability parameters
	GradClip    float64 // <=0 disables (default 1.0 is a good start)
	WeightDecay float64 // AdamW-style, e.g., 0.01; 0 disables
	Debug       bool    // enable periodic debug logs
	DebugEvery  int     // print every N optimizer steps
}

var layers = 6
var config = TrainingConfig{
	DModel:     512,
	HiddenSize: 1024,
	VocabSize:  4096, // Top number of 1-4 chars
	NumHeads:   8,    // dHead = DModel/NumHeads
	SeqLen:     64,   // max context
	AttnLR:     0.0003,
	MLPLR:      0.0003,
	UnembedLR:  0.00003,
	NormLR:     0.0003,

	MaxEpochs: 1000,
	Patience:  50,
	Epsilon:   1e-4,
	BatchSize: 1024, // each example is one prefix
	ValFrac:   0.1,

	WarmupSteps: 10_000,
	DecaySteps:  1_000_000,
	AdamBeta1:   0.9,
	AdamBeta2:   0.999,
	AdamEps:     1e-8,

	GradClip:    1.0,
	WeightDecay: 0.01,
	Debug:       false,
	DebugEvery:  1000,
}

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()

	gpt := CreateGPT(
		config.DModel,
		config.HiddenSize,
		config.VocabSize,
		config.AttnLR,
		config.MLPLR,
	)

	// Build vocab + embeddings via streaming pass (no full dataset in memory)
	linesCount, err := buildVocabAndEmbFromTrain(gpt)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Initialized vocab (%d) and embeddings from training file. Estimated lines: %d\n",
		len(vocab.IDToToken), linesCount)

	trainPath := findTrainFile()
	iter, err := newTrainLineIter(trainPath)
	if err != nil {
		fmt.Println("failed to open training file:", err)
		return
	}
	defer iter.close()

	dYbuf := mat.NewDense(config.DModel, config.SeqLen, nil)
	var bestAccuracy float64 = -1.0
	var noImprovementCount int
	bestModel := gpt

	fmt.Printf("Train (streaming): lines≈%d  Eval: from eval.en\n", linesCount)

	adamStep := 0

	for e := 0; e < config.MaxEpochs; e++ {
		var totalLoss float64
		var steps float64

		epochTime := time.Now()

		// Random-sample BatchSize prefix examples; build X and target on the fly.
		B := min(config.BatchSize, 1000000000) // cap for safety
		for b := 0; b < B; b++ {
			// pick a random sequence and position
			ids, err := iter.nextIDs()
			if err != nil {
				// Reached EOF; rewind happened. Fetch again.
				ids, err = iter.nextIDs()
				if err != nil {
					continue
				}
			}

			if len(ids) < 2 {
				continue
			}

			i := rand.Intn(len(ids) - 1) // predict ids[i+1]
			start := 0
			if i+1 > config.SeqLen {
				start = i + 1 - config.SeqLen
			}
			Xctx := embedSequence(emb, ids[start:i+1]) // (d x T)
			target := oneHot(len(vocab.IDToToken), ids[i+1])

			Y := Xctx
			for i := 0; i < layers; i++ {
				Y = gpt.blocks[i].Forward(Y)
			}

			// take last position only
			yLast := lastCol(Y)
			logits := Unembed(yLast)

			// Loss + gradient
			loss, _ := CrossEntropyWithGrad(logits, target)

			adamStep++
			attnLR := LRSchedule(adamStep, config.AttnLR)
			mlpLR := LRSchedule(adamStep, config.MLPLR)
			normLR := LRSchedule(adamStep, config.NormLR)
			unembedLR := LRSchedule(adamStep, config.UnembedLR)

			// Freeze embeddings during early warmup to avoid regressions
			if adamStep < config.WarmupSteps/2 {
				unembedLR = 0
			}

			for i := 0; i < layers; i++ {
				gpt.blocks[i].attn.learningRate = attnLR
				gpt.blocks[i].mlp.learningRate = mlpLR
				gpt.blocks[i].ln1.learningRate, gpt.blocks[i].ln2.learningRate = normLR, normLR
			}

			totalLoss += loss
			steps++

	// -------- FULL-SEQUENCE LOSS --------
    cols := Xctx.RawMatrix().Cols
    dY := dYbuf.Slice(0, config.DModel, 0, cols).(*mat.Dense)
    // zero dY
    r, c := dY.Dims()
    for i := 0; i < r; i++ {
        for j := 0; j < c; j++ {
            dY.Set(i, j, 0)
        }
    }

    // accumulate loss and grads for every timestep
    totalLoss := 0.0
    dEmb := mat.NewDense(emb.RawMatrix().Rows, emb.RawMatrix().Cols, nil)
    for t := 0; t < cols-1; t++ {
        // logits at time t: (vocab x 1) = emb * y[:,t]
        yCol := Y.Slice(0, config.DModel, t, t+1).(*mat.Dense)
        logits := toDense(dot(emb.T(), yCol)) // (VocabSize x 1)

        // target is token at position t+1
        goldID := ids[t+1]
        target := oneHot(config.VocabSize, goldID)

        loss, gradLogits := CrossEntropyWithGrad(logits, target)
        totalLoss += loss

        // dY[:,t] = emb^T * (p - t)
        dyCol := toDense(dot(emb, gradLogits)) // (DModel x 1)
        for i := 0; i < config.DModel; i++ {
            dY.Set(i, t, dyCol.At(i, 0))
        }

        // accumulate dEmb += (p - t) * yCol^T
        dEmb.Add(dEmb, toDense(dot(yCol, gradLogits.T())))
    }

    // update embeddings with AdamW
    initEmbAdamIfNeeded()
    embT++
    if config.GradClip > 0 {
        clipGrads(config.GradClip, dEmb)
    }
    adamUpdateInPlace(emb, dEmb, embM, embV, embT,
        unembedLR, config.AdamBeta1, config.AdamBeta2, config.AdamEps,
        config.WeightDecay)

			for i := layers - 1; i >= 0; i-- {
				dY = gpt.blocks[i].Backward(dY)
			}
		}

		// Calculate average loss for the epoch
		avgLoss := totalLoss / steps

		// Evaluate accuracy on the test set
		correct, total := evaluateAccuracy(gpt)
		var accuracy float64
		if total > 0 {
			accuracy = float64(correct) / float64(total)
		} else {
			accuracy = 0
		}

		elapsed := time.Since(epochTime)
		fmt.Printf("Epoch %d - Accuracy: %.4f, Loss: %.4f, Time for epoch: %s\n", e+1, accuracy, avgLoss, elapsed)
		fmt.Printf("Before epoch %d: Attn.Wq[0] norm=%.6g MLP.hidden norm=%.6g\n",
			e+1,
			matrixNorm(gpt.blocks[0].attn.Wquery[0]),
			matrixNorm(gpt.blocks[0].mlp.hiddenWeights),
		)

		// --- Early stopping logic based on loss improvement and accuracy checkpointing ---
		// Check if the current accuracy is the best we've seen so far.
		if accuracy > bestAccuracy && e > 20 {
			bestAccuracy = accuracy
			// Deep copy weights
			tmpFile := "models/tmp_best.gob"
			if err := SaveTransformer(&gpt, tmpFile); err == nil {
				var clone Transformer = CreateGPT(config.DModel, config.HiddenSize, config.VocabSize, config.AttnLR, config.MLPLR)
				if err := LoadTransformer(&clone, tmpFile); err == nil {
					bestModel = clone
				}
				_ = os.Remove(tmpFile)
			}
			noImprovementCount = 0
		} else {
			noImprovementCount++
		}

		// The loop now breaks if we've seen enough epochs without a new best accuracy.
		if noImprovementCount >= config.Patience {
			fmt.Println("\nStopping training early due to lack of improvement in accuracy.")
			break
		}
		// If the loss func is too small, stop training.
		if avgLoss < config.Epsilon {
			fmt.Println("\nStopping training early due to loss being too small.")
			break
		}

		// --- End Early Stopping Logic ---
	}

	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to train: %s\n", elapsed)

	// After the training loop, save the best-performing model that was found.
	if err := save(bestModel); err != nil {
		fmt.Println("Error saving model:", err)
		return
	} else {
		fmt.Println("Saved the best performing model.")
		chatCLI(&gpt)
	}
}
