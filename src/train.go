package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/manningwu07/GPT/IO"
	"github.com/manningwu07/GPT/optimizations"
	"github.com/manningwu07/GPT/params"
	"github.com/manningwu07/GPT/transformer"
	"github.com/manningwu07/GPT/utils"
	"gonum.org/v1/gonum/mat"
)

func TrainGPT(gpt *transformer.Transformer, iter *IO.TrainLineIter, linesCount int) *transformer.Transformer {
	dYbuf := mat.NewDense(params.Config.DModel, params.Config.SeqLen, nil)
	var bestAccuracy float64 = -1.0
	var noImprovementCount int
	bestModel := *gpt // copy initial model

	fmt.Printf("Train (streaming): lines≈%d  Eval: from eval.en\n", linesCount)

	adamStep := 0

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

			Y := Xctx
			for i := 0; i < params.Layers; i++ {
				Y = gpt.Blocks[i].Forward(Y)
			}

			// take last position only
			yLast := utils.LastCol(Y)
			logits := IO.Unembed(yLast)

			// Loss + gradient
			loss, _ := utils.CrossEntropyWithIndex(logits, ids[i+1])

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

				loss, gradLogits := utils.CrossEntropyWithIndex(logits, goldID)

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
				0.0)
		}

		// Calculate average loss for the epoch

		avgTokLoss := 0.0
		trainPPL := 0.0
		if tokenCounter > 0 {
			avgTokLoss = totalTokenLoss / float64(tokenCounter)
			trainPPL = math.Exp(avgTokLoss)
		}

		// Evaluate accuracy on the test set
		corr, tot, ceSum := IO.EvaluateMetrics(*gpt)
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

		// --- Early stopping logic based on loss improvement and accuracy checkpointing ---
		// Check if the current accuracy is the best we've seen so far.
		alreadySaved := false
		if accuracy > bestAccuracy+ params.Config.ImprovementThreshold && e > 20 {
			bestAccuracy = accuracy
			_ = transformer.SaveTransformer(gpt, "models/best_model.gob")
			bestModel = *gpt
			noImprovementCount = 0
			alreadySaved = true
		} else {
			noImprovementCount++
		}

		// Saves every X Epochs
		if (e+1)%params.Config.SaveEpochNumber == 0 && !alreadySaved {
			_ = safeSaveTransformer(gpt, fmt.Sprintf("models/last_epoch.gob"))
			fmt.Printf("Saved checkpoint at epoch %d\n", e+1)
			alreadySaved = false
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

	return &bestModel
}
