package main

import (
	"encoding/csv"
	"os"
	"time"

	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// How many times does attn --> mlp happen
var layers = 8

type TrainingConfig struct {
	DModel     int // model width
	HiddenSize int // MLP hidden
	VocabSize  int // |V|
	NumHeads   int // attention heads
	SeqLen     int // max prefix length (context length)
	AttnLR     float64
	MLPLR      float64
	UnembedLR  float64

	MaxEpochs int     // maximum number of epochs
	Patience  int     // early stopping patience
	Epsilon   float64 // stop if loss < epsilon
	BatchSize int     // mini-batch size
	ValFrac   float64 // fraction of data held out for validation
}

// Reasonable defaults for small experiments
var config = TrainingConfig{
	DModel:     256, 
	HiddenSize: 512, 
	VocabSize:  32768, // Top number of 1-4 chars
	NumHeads:   4,    // dHead = DModel/NumHeads
	SeqLen:     128,  // max context
	AttnLR:     0.003, // simple SGD -> smaller LRs
	MLPLR:      0.003,
	UnembedLR:  0.003,

	MaxEpochs: 25,
	Patience:  10,
	Epsilon:   1e-4,
	BatchSize: 1024, // each example is one prefix
	ValFrac:   0.1,
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

	// Load all training data (May need to change as memory becomes an issue for bigger training data)
	trainSeqs, err := loadTrainSequences(gpt)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Loaded %d training sequences.\n", len(trainSeqs))

	var bestAccuracy float64 = -1.0
	var noImprovementCount int
	bestModel := gpt

	fmt.Printf("Train sequences: %d  Eval: from eval.en\n", len(trainSeqs))

	// Create or truncate the log file
	logFile, err := os.Create("training_log.csv")
	if err != nil {
		fmt.Println("Error creating log file:", err)
		return
	}
	defer logFile.Close()
	logWriter := csv.NewWriter(logFile)
	logWriter.Write([]string{"epoch", "accuracy", "loss"})
	defer logWriter.Flush()

	for e := 0; e < config.MaxEpochs; e++ {
		var totalLoss float64
		var steps float64

		epochTime := time.Now()

		// Random-sample BatchSize prefix examples; build X and target on the fly.
		B := min(config.BatchSize, 1000000000) // cap for safety
		for b := 0; b < B; b++ {
			// pick a random sequence and position
			si := rand.Intn(len(trainSeqs))
			ids := trainSeqs[si]
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
			loss, gradLogits := CrossEntropyWithGrad(logits, target)

			totalLoss += loss
			steps++

			// Backprop through tied unembedding: logits = emb^T * yLast
			// dyLast = emb * (p - t)
			dyLast := toDense(dot(emb, gradLogits))
			// dEmb = yLast * (p - t)^T
			dEmb := toDense(dot(yLast, gradLogits.T()))
			emb = toDense(add(emb, scale(-config.UnembedLR, dEmb)))

			dY := mat.NewDense(config.DModel, Xctx.RawMatrix().Cols, nil)
			for i := 0; i < config.DModel; i++ {
				dY.Set(i, dY.RawMatrix().Cols-1, dyLast.At(i, 0))
			}
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
		if accuracy > bestAccuracy && e > 5 { // Second condition is to prevent the gpt from being cloned during the beginning
			bestAccuracy = accuracy
			bestModel = gpt
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
	} else {
		fmt.Println("Saved the best performing model.")
	}
}
