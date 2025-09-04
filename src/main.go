package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"os/signal"
	"syscall"

	"github.com/manningwu07/GPT/IO"
	"github.com/manningwu07/GPT/optimizations"
	"github.com/manningwu07/GPT/params"
	"github.com/manningwu07/GPT/transformer"
	"github.com/manningwu07/GPT/utils"
	"gonum.org/v1/gonum/mat"
)

// How many times does attn --> mlp happen

var (
	resumePath string
	debugflag  bool
)

func init() {
	flag.StringVar(&resumePath, "resume", "", "Path to checkpoint .gob file to resume from")
	flag.BoolVar(&debugflag, "debug", false, "Enable debug logging")
}

func main() {
	flag.Parse()

	if debugflag {
		params.Config.Debug = true
	}

	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()

	var gpt transformer.Transformer
	if resumePath != "" {
		fmt.Printf("Resuming from checkpoint: %s\n", resumePath)
		gpt = transformer.CreateGPT(
			params.Config.DModel,
			params.Config.HiddenSize,
			params.Config.VocabSize,
			params.Config.AttnLR,
			params.Config.MLPLR,
		)
		if err := transformer.LoadTransformer(&gpt, resumePath); err != nil {
			fmt.Println("Failed to load checkpoint:", err)
			os.Exit(1)
		}
	} else {
		fmt.Println("Starting new model from scratch")
		gpt = transformer.CreateGPT(
			params.Config.DModel,
			params.Config.HiddenSize,
			params.Config.VocabSize,
			params.Config.AttnLR,
			params.Config.MLPLR,
		)
	}

	installSignalCheckpoints(&gpt)

	installSignalCheckpoints(&gpt)

	// Build vocab + embeddings via streaming pass (no full dataset in memory)
	linesCount, err := IO.BuildVocabAndEmbFromTrain(gpt)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Initialized vocab (%d) and embeddings from training file. Estimated lines: %d\n",
		len(params.Vocab.IDToToken), linesCount)

	// Tiny overfit mode
	if os.Getenv("OVERFIT_TINY") == "1" {
		overfitTiny(&gpt, 100, 5000) // 100 lines, 1000 steps
		ChatCLI(&gpt)
		return // exit after tiny test
	}

	trainPath := IO.FindTrainFile()
	iter, err := IO.NewTrainLineIter(trainPath)
	if err != nil {
		fmt.Println("failed to open training file:", err)
		return
	}
	defer iter.Close()

	dYbuf := mat.NewDense(params.Config.DModel, params.Config.SeqLen, nil)
	var bestAccuracy float64 = -1.0
	var noImprovementCount int
	bestModel := gpt

	fmt.Printf("Train (streaming): lines≈%d  Eval: from eval.en\n", linesCount)

	adamStep := 0
	lastSave := time.Now()

	for e := 0; e < params.Config.MaxEpochs; e++ {
		var totalLoss float64
		var steps float64
		var tokenCounter int
		var totalTokenLoss float64

		start := time.Now()

		// Random-sample BatchSize prefix examples; build X and target on the fly.
		B := min(params.Config.BatchSize, 1000000000) // cap for safety
		for b := 0; b < B; b++ {
			// pick a random sequence and position
			ids, err := iter.NextIDs()
			if err != nil {
				// Reached EOF; rewind happened. Fetch again.
				ids, err = iter.NextIDs()
				if err != nil {
					continue
				}
			}

			if len(ids) < 2 {
				continue
			}

			i := rand.Intn(len(ids) - 1) // predict ids[i+1]
			start := 0
			if i+1 > params.Config.SeqLen {
				start = i + 1 - params.Config.SeqLen
			}
			Xctx := IO.EmbedSequence(params.Emb, ids[start:i+1]) // (d x T)
			target := utils.OneHot(len(params.Vocab.IDToToken), ids[i+1])

			Y := Xctx
			for i := 0; i < params.Layers; i++ {
				Y = gpt.Blocks[i].Forward(Y)
			}

			// take last position only
			yLast := utils.LastCol(Y)
			logits := IO.Unembed(yLast)

			// Loss + gradient
			loss, _ := utils.CrossEntropyWithGrad(logits, target)

			adamStep++
			attnLR := utils.LRSchedule(adamStep, params.Config.AttnLR)
			mlpLR := utils.LRSchedule(adamStep, params.Config.MLPLR)
			normLR := utils.LRSchedule(adamStep, params.Config.NormLR)
			unembedLR := utils.LRSchedule(adamStep, params.Config.UnembedLR)

			// Freeze embeddings during early warmup to avoid regressions
			if adamStep < params.Config.WarmupSteps/2 {
				unembedLR = 0
			}

			for i := 0; i < params.Layers; i++ {
				gpt.Blocks[i].Attn.LearningRate = attnLR
				gpt.Blocks[i].Mlp.LearningRate = mlpLR
				gpt.Blocks[i].Ln1.LearningRate, gpt.Blocks[i].Ln2.LearningRate = normLR, normLR
			}

			totalLoss += loss
			steps++

			// -------- FULL-SEQUENCE LOSS --------
			cols := Xctx.RawMatrix().Cols
			dY := dYbuf.Slice(0, params.Config.DModel, 0, cols).(*mat.Dense)
			// zero dY
			r, c := dY.Dims()
			for i := 0; i < r; i++ {
				for j := 0; j < c; j++ {
					dY.Set(i, j, 0)
				}
			}

			// accumulate loss and grads for every timestep
			seqTokLoss := 0.0
			dEmb := mat.NewDense(params.Emb.RawMatrix().Rows, params.Emb.RawMatrix().Cols, nil)
			for t := 0; t < cols-1; t++ {
				// logits at time t: (vocab x 1) = emb * y[:,t]
				yCol := Y.Slice(0, params.Config.DModel, t, t+1).(*mat.Dense)
				logits := utils.ToDense(utils.Dot(params.Emb.T(), yCol)) // (VocabSize x 1)

				// target is token at position t+1
				goldID := ids[t+1]
				target := utils.OneHot(params.Config.VocabSize, goldID)

				loss, gradLogits := utils.CrossEntropyWithGrad(logits, target)

				// dY[:,t] = emb^T * (p - t)
				dyCol := utils.ToDense(utils.Dot(params.Emb, gradLogits)) // (DModel x 1)
				for i := 0; i < params.Config.DModel; i++ {
					dY.Set(i, t, dyCol.At(i, 0))
				}

				// accumulate dEmb += (p - t) * yCol^T
				dEmb.Add(dEmb, utils.ToDense(utils.Dot(yCol, gradLogits.T())))
				seqTokLoss += loss
			}

			totalTokenLoss += seqTokLoss
			tokenCounter += (cols - 1) // number of tokens predicted in this sequence
			totalLoss += seqTokLoss

			for i := params.Layers - 1; i >= 0; i-- {
				dY = gpt.Blocks[i].Backward(dY)
			}

			dXinput := dY // (dModel x T), gradient wrt Xctx (which is emb + pos)

			// Accumulate input-embedding gradient into emb columns used at each t
			dEmbIn := mat.NewDense(params.Emb.RawMatrix().Rows, params.Emb.RawMatrix().Cols, nil)
			for t := 0; t < cols; t++ {
				if start+t >= len(ids) {
					break
				}
				tokID := ids[start+t]
				// add dXinput[:,t] into column tokID of dEmbIn
				colGrad := dXinput.Slice(0, params.Config.DModel, t, t+1).(*mat.Dense)
				// write-add into dEmbIn[:, tokID]
				for i := 0; i < params.Config.DModel; i++ {
					dEmbIn.Set(i, tokID, dEmbIn.At(i, tokID)+colGrad.At(i, 0))
				}
			}
			// Add input path grad to output path grad
			dEmb.Add(dEmb, dEmbIn)

			// Positional embedding gradient: since X = emb + pos, dPos[:,t] += dXinput[:,t]
			optimizations.InitPosAdamIfNeeded()
			dPos := mat.NewDense(params.PosEmb.RawMatrix().Rows, params.PosEmb.RawMatrix().Cols, nil)
			maxT := cols
			if maxT > params.PosEmb.RawMatrix().Cols {
				maxT = params.PosEmb.RawMatrix().Cols
			}
			for t := 0; t < maxT; t++ {
				for i := 0; i < params.Config.DModel; i++ {
					dPos.Set(i, t, dPos.At(i, t)+dXinput.At(i, t))
				}
			}

			// Now update embeddings and positional embeddings with AdamW
			optimizations.InitEmbAdamIfNeeded()
			params.EmbT++
			params.PosT++
			if params.Config.GradClip > 0 {
				utils.ClipGrads(params.Config.GradClip, dEmb, dPos)
			}
			optimizations.AdamUpdateInPlace(params.Emb, dEmb, params.EmbM, params.EmbV, params.EmbT,
				unembedLR, params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps,
				params.Config.WeightDecay)
			optimizations.AdamUpdateInPlace(params.PosEmb, dPos, params.PosM, params.PosV, params.PosT,
				params.Config.PosLR, params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps,
				0.0) // typically no weight decay for pos embeddings

			// Optional periodic checkpoint by step/time
			if params.Config.SaveEverySteps > 0 && adamStep%params.Config.SaveEverySteps == 0 {
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
		corr, tot, ceSum := IO.EvaluateMetrics(gpt)
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
			utils.MatrixNorm(gpt.Blocks[0].Attn.Wquery[0]),
			utils.MatrixNorm(gpt.Blocks[0].Mlp.HiddenWeights),
		)

		_ = safeSaveTransformer(&gpt, "models/last_epoch.gob")
		lastSave = time.Now()

		// --- Early stopping logic based on loss improvement and accuracy checkpointing ---
		// Check if the current accuracy is the best we've seen so far.
		if accuracy > bestAccuracy && e > 20 {
			bestAccuracy = accuracy
			_ = transformer.SaveTransformer(&gpt, "models/best_model.gob")
			noImprovementCount = 0
		} else {
			noImprovementCount++
		}

		// The loop now breaks if we've seen enough epochs without a new best accuracy.
		if noImprovementCount >= params.Config.Patience {
			fmt.Println("\nStopping training early due to lack of improvement in accuracy.")
			break
		}

		// If the loss func is too small, stop training.
		avgLoss := totalLoss / steps
		if avgLoss < params.Config.Epsilon {
			fmt.Println("\nStopping training early due to loss being too small.")
			break
		}

		// --- End Early Stopping Logic ---
	}

	elapsed := time.Since(t1)
	fmt.Printf("\nTime taken to train: %s\n", elapsed)

	// After the training loop, save the best-performing model that was found.
	if err := transformer.Save(bestModel); err != nil {
		fmt.Println("Error saving model:", err)
		return
	} else {
		fmt.Println("Saved the best performing model.")
		ChatCLI(&gpt)
	}
}

// Tiny overfit mode: train on first N lines for S steps to sanity-check the loop.
// Enable by setting OVERFIT_TINY=1 in the environment.
func overfitTiny(gpt *transformer.Transformer, N, steps int) {
	seqs, err := IO.LoadTinyTrainIDs(N)
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
		X := IO.EmbedSequence(params.Emb, ids[:len(ids)-1])
		// forward
		Y := X
		for l := 0; l < params.Layers; l++ {
			Y = gpt.Blocks[l].Forward(Y)
		}
		cols := Y.RawMatrix().Cols
		dY := mat.NewDense(params.Config.DModel, cols, nil)
		dEmb := mat.NewDense(params.Emb.RawMatrix().Rows, params.Emb.RawMatrix().Cols, nil)
		seqCE := 0.0
		for t := 0; t < cols; t++ {
			yCol := Y.Slice(0, params.Config.DModel, t, t+1).(*mat.Dense)
			logits := utils.ToDense(utils.Dot(params.Emb.T(), yCol))
			goldID := ids[t+1] // next token
			oh := utils.OneHot(params.Config.VocabSize, goldID)
			loss, gradLogits := utils.CrossEntropyWithGrad(logits, oh)
			seqCE += loss
			dyCol := utils.ToDense(utils.Dot(params.Emb, gradLogits)) // (DModel x 1)
			for i := 0; i < params.Config.DModel; i++ {
				dY.Set(i, t, dyCol.At(i, 0))
			}
			dEmb.Add(dEmb, utils.ToDense(utils.Dot(yCol, gradLogits.T())))
		}
		// update embeddings
		optimizations.InitEmbAdamIfNeeded()
		params.EmbT++
		if params.Config.GradClip > 0 {
			utils.ClipGrads(params.Config.GradClip, dEmb)
		}
		optimizations.AdamUpdateInPlace(
			params.Emb, dEmb, params.EmbM, params.EmbV, params.EmbT,
			params.Config.UnembedLR, params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps,
			params.Config.WeightDecay,
		)
		// backprop through blocks
		for l := params.Layers - 1; l >= 0; l-- {
			dY = gpt.Blocks[l].Backward(dY)
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
func installSignalCheckpoints(gpt *transformer.Transformer) {
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
func safeSaveTransformer(gpt *transformer.Transformer, path string) error {
	_ = os.MkdirAll("models", 0o755)
	tmp := path + ".tmp"
	if err := transformer.SaveTransformer(gpt, tmp); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}