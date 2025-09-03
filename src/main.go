package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"
	"flag"

	"gonum.org/v1/gonum/mat"
	"os/signal"
    "syscall"
)

// How many times does attn --> mlp happen

var (
    resumePath string
    debugFlag  bool
)

func init() {
    flag.StringVar(&resumePath, "resume", "", "Path to checkpoint .gob file to resume from")
    flag.BoolVar(&debugFlag, "debug", false, "Enable debug logging")
}

// Adam Optimizer global vars
var (
	embM, embV *mat.Dense
	embT       int
	posEmb *mat.Dense
	posM, posV *mat.Dense
	posT int
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
	PosLR       float64 // learning rate for positional embeddings
    SaveEverySteps int  // checkpoint every N optimizer steps (0=disable)
}

var layers = 6
var config = TrainingConfig{
	DModel:     512,
	HiddenSize: 1024,
	VocabSize:  8192, // Top number of 1-4 chars
	NumHeads:   8,    // dHead = DModel/NumHeads
	SeqLen:     64,   // max context
	AttnLR:     0.0003,
	MLPLR:      0.0003,
	UnembedLR:  0.00003,
	NormLR:     0.0003,

	MaxEpochs: 250,
	Patience:  25,
	Epsilon:   1e-4,
	BatchSize: 2048, // each example is one prefix
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
	PosLR:       0.0003,
    SaveEverySteps: 10000,
}

func main() {
	flag.Parse()

    if debugFlag {
        config.Debug = true
    }

	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()

	var gpt Transformer
    if resumePath != "" {
        fmt.Printf("Resuming from checkpoint: %s\n", resumePath)
        gpt = CreateGPT(
            config.DModel,
            config.HiddenSize,
            config.VocabSize,
            config.AttnLR,
            config.MLPLR,
        )
        if err := LoadTransformer(&gpt, resumePath); err != nil {
            fmt.Println("Failed to load checkpoint:", err)
            os.Exit(1)
        }
    } else {
        fmt.Println("Starting new model from scratch")
        gpt = CreateGPT(
            config.DModel,
            config.HiddenSize,
            config.VocabSize,
            config.AttnLR,
            config.MLPLR,
        )
    }

    installSignalCheckpoints(&gpt)

	installSignalCheckpoints(&gpt)

	// Build vocab + embeddings via streaming pass (no full dataset in memory)
	linesCount, err := buildVocabAndEmbFromTrain(gpt)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Initialized vocab (%d) and embeddings from training file. Estimated lines: %d\n",
		len(vocab.IDToToken), linesCount)

	
	// Tiny overfit mode
	if os.Getenv("OVERFIT_TINY") == "1" {
        overfitTiny(&gpt, 100, 5000) // 100 lines, 1000 steps
		chatCLI(&gpt)
        return                       // exit after tiny test
    }

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
	lastSave := time.Now()

	for e := 0; e < config.MaxEpochs; e++ {
		var totalLoss float64
		var steps float64
		var tokenCounter int
		var totalTokenLoss float64

		start := time.Now()

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
			seqTokLoss := 0.0
			dEmb := mat.NewDense(emb.RawMatrix().Rows, emb.RawMatrix().Cols, nil)
			for t := 0; t < cols-1; t++ {
				// logits at time t: (vocab x 1) = emb * y[:,t]
				yCol := Y.Slice(0, config.DModel, t, t+1).(*mat.Dense)
				logits := toDense(dot(emb.T(), yCol)) // (VocabSize x 1)

				// target is token at position t+1
				goldID := ids[t+1]
				target := oneHot(config.VocabSize, goldID)

				loss, gradLogits := CrossEntropyWithGrad(logits, target)

				// dY[:,t] = emb^T * (p - t)
				dyCol := toDense(dot(emb, gradLogits)) // (DModel x 1)
				for i := 0; i < config.DModel; i++ {
					dY.Set(i, t, dyCol.At(i, 0))
				}

				// accumulate dEmb += (p - t) * yCol^T
				dEmb.Add(dEmb, toDense(dot(yCol, gradLogits.T())))
				seqTokLoss += loss
			}

			totalTokenLoss += seqTokLoss
			tokenCounter += (cols - 1) // number of tokens predicted in this sequence
			totalLoss += seqTokLoss

			for i := layers - 1; i >= 0; i-- {
				dY = gpt.blocks[i].Backward(dY)
			}

			dXinput := dY // (dModel x T), gradient wrt Xctx (which is emb + pos)

            // Accumulate input-embedding gradient into emb columns used at each t
            dEmbIn := mat.NewDense(emb.RawMatrix().Rows, emb.RawMatrix().Cols, nil)
            for t := 0; t < cols; t++ {
                if start+t >= len(ids) { break }
                tokID := ids[start+t]
                // add dXinput[:,t] into column tokID of dEmbIn
                colGrad := dXinput.Slice(0, config.DModel, t, t+1).(*mat.Dense)
                // write-add into dEmbIn[:, tokID]
                for i := 0; i < config.DModel; i++ {
                    dEmbIn.Set(i, tokID, dEmbIn.At(i, tokID)+colGrad.At(i, 0))
                }
            }
            // Add input path grad to output path grad
            dEmb.Add(dEmb, dEmbIn)

            // Positional embedding gradient: since X = emb + pos, dPos[:,t] += dXinput[:,t]
            initPosAdamIfNeeded()
            dPos := mat.NewDense(posEmb.RawMatrix().Rows, posEmb.RawMatrix().Cols, nil)
            maxT := cols
            if maxT > posEmb.RawMatrix().Cols { maxT = posEmb.RawMatrix().Cols }
            for t := 0; t < maxT; t++ {
                for i := 0; i < config.DModel; i++ {
                    dPos.Set(i, t, dPos.At(i, t)+dXinput.At(i, t))
                }
            }

            // Now update embeddings and positional embeddings with AdamW
            initEmbAdamIfNeeded()
            embT++
            posT++
            if config.GradClip > 0 {
                clipGrads(config.GradClip, dEmb, dPos)
            }
            adamUpdateInPlace(emb, dEmb, embM, embV, embT,
                unembedLR, config.AdamBeta1, config.AdamBeta2, config.AdamEps,
                config.WeightDecay)
            adamUpdateInPlace(posEmb, dPos, posM, posV, posT,
                config.PosLR, config.AdamBeta1, config.AdamBeta2, config.AdamEps,
                0.0) // typically no weight decay for pos embeddings

            // Optional periodic checkpoint by step/time
            if config.SaveEverySteps > 0 && adamStep%config.SaveEverySteps == 0 {
                _ = safeSaveTransformer(&gpt, "models/ckpt_latest.gob")
            }
            if time.Since(lastSave) > 10*time.Minute {
                _ = safeSaveTransformer(&gpt, "models/ckpt_latest.gob")
				lastSave = time.Now()
            }
		}

		// Calculate average loss for the epoch

		avgTokLoss := 0.0
		trainPPL := 0.0
		if tokenCounter > 0 {
			avgTokLoss = totalTokenLoss / float64(tokenCounter)
			trainPPL = math.Exp(avgTokLoss)
		}

		// Evaluate accuracy on the test set
		corr, tot, ceSum := evaluateMetrics(gpt)
		accuracy := 0.0
		evalPPL := 0.0
		if tot > 0 {
			accuracy = float64(corr) / float64(tot)
			evalPPL = math.Exp(ceSum / float64(tot))
		}
		fmt.Printf(
			"Epoch %d - Acc: %.4f, TrainTokLoss: %.4f, TrainPPL: %.1f, EvalPPL: %.1f, Time: %v\n",
			e, accuracy, avgTokLoss, trainPPL, evalPPL, time.Since(start),
		)
		fmt.Printf("Before epoch %d: Attn.Wq[0] norm=%.6g MLP.hidden norm=%.6g\n",
			e+1,
			matrixNorm(gpt.blocks[0].attn.Wquery[0]),
			matrixNorm(gpt.blocks[0].mlp.hiddenWeights),
		)

		_ = safeSaveTransformer(&gpt, "models/last_epoch.gob")
        lastSave = time.Now()

		// --- Early stopping logic based on loss improvement and accuracy checkpointing ---
		// Check if the current accuracy is the best we've seen so far.
		if accuracy > bestAccuracy && e > 20 {
			bestAccuracy = accuracy
			_ = SaveTransformer(&gpt, "models/best_model.gob")
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
		avgLoss := totalLoss / steps
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

// Tiny overfit mode: train on first N lines for S steps to sanity-check the loop.
// Enable by setting OVERFIT_TINY=1 in the environment.
func overfitTiny(gpt *Transformer, N, steps int) {
	seqs, err := loadTinyTrainIDs(N)
	if err != nil || len(seqs) == 0 {
		fmt.Println("overfitTiny: no data:", err)
		return
	}
	fmt.Printf("OverfitTiny: %d sequences, %d steps\n", len(seqs), steps)
	totalCE := 0.0
	totalTok := 0
	for step := 0; step < steps; step++ {
		ids := seqs[step%len(seqs)]
		// inputs are all but last token
		X := embedSequence(emb, ids[:len(ids)-1])
		// forward
		Y := X
		for l := 0; l < layers; l++ {
			Y = gpt.blocks[l].Forward(Y)
		}
		cols := Y.RawMatrix().Cols
		dY := mat.NewDense(config.DModel, cols, nil)
		dEmb := mat.NewDense(emb.RawMatrix().Rows, emb.RawMatrix().Cols, nil)
		seqCE := 0.0
		for t := 0; t < cols; t++ {
			yCol := Y.Slice(0, config.DModel, t, t+1).(*mat.Dense)
			logits := toDense(dot(emb.T(), yCol))
			goldID := ids[t+1] // next token
			oh := oneHot(config.VocabSize, goldID)
			loss, gradLogits := CrossEntropyWithGrad(logits, oh)
			seqCE += loss
			dyCol := toDense(dot(emb, gradLogits)) // (DModel x 1)
			for i := 0; i < config.DModel; i++ {
				dY.Set(i, t, dyCol.At(i, 0))
			}
			dEmb.Add(dEmb, toDense(dot(yCol, gradLogits.T())))
		}
		// update embeddings
		initEmbAdamIfNeeded()
		embT++
		if config.GradClip > 0 {
			clipGrads(config.GradClip, dEmb)
		}
		adamUpdateInPlace(
			emb, dEmb, embM, embV, embT,
			config.UnembedLR, config.AdamBeta1, config.AdamBeta2, config.AdamEps,
			config.WeightDecay,
		)
		// backprop through blocks
		for l := layers - 1; l >= 0; l-- {
			dY = gpt.blocks[l].Backward(dY)
		}
		totalCE += seqCE
		totalTok += cols
		if (step+1)%100 == 0 {
			avgTokLoss := totalCE / float64(totalTok)
			fmt.Printf("Tiny step %4d: tokLoss=%.4f ppl=%.1f\n",
				step+1, avgTokLoss, math.Exp(avgTokLoss))
		}
	}
	avgTokLoss := totalCE / float64(totalTok)
	fmt.Printf("OverfitTiny done: tokLoss=%.4f ppl=%.1f\n",
		avgTokLoss, math.Exp(avgTokLoss))
}


// installSignalCheckpoints installs handlers:
// - SIGUSR1: save checkpoint and continue
// - SIGINT/SIGTERM: save checkpoint then exit
func installSignalCheckpoints(gpt *Transformer) {
    ch := make(chan os.Signal, 2)
    signal.Notify(ch, os.Interrupt, syscall.SIGTERM, syscall.SIGUSR1)
    go func() {
        for sig := range ch {
            switch sig {
            case syscall.SIGUSR1:
                fmt.Println("\n[signal] SIGUSR1 received: saving checkpoint...")
                if err := safeSaveTransformer(gpt, "models/ckpt_latest.gob"); err != nil {
                    fmt.Println("checkpoint save error:", err)
                } else {
                    fmt.Println("checkpoint saved to models/ckpt_latest.gob")
                }
            case os.Interrupt, syscall.SIGTERM:
                fmt.Println("\n[signal] Interrupt/TERM received: saving checkpoint and exiting...")
                _ = safeSaveTransformer(gpt, "models/ckpt_latest.gob")
                os.Exit(0)
            }
        }
    }()
}

// safeSaveTransformer writes to a temp file then renames atomically.
func safeSaveTransformer(gpt *Transformer, path string) error {
    _ = os.MkdirAll("models", 0o755)
    tmp := path + ".tmp"
    if err := SaveTransformer(gpt, tmp); err != nil {
        _ = os.Remove(tmp)
        return err
    }
    return os.Rename(tmp, path)
}