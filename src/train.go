package main

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/manningwu07/GPT/IO"
	"github.com/manningwu07/GPT/optimizations"
	"github.com/manningwu07/GPT/params"
	"github.com/manningwu07/GPT/transformer"
	"github.com/manningwu07/GPT/utils"
	"gonum.org/v1/gonum/mat"
)

type sample struct {
	ids   []int
	i     int
	start int
}

func TrainGPT(gpt *transformer.Transformer, iter *IO.TrainLineIter, linesCount int) *transformer.Transformer {
	dYbuf := mat.NewDense(params.Config.DModel, params.Config.SeqLen, nil)
	var bestAccuracy float64 = -1.0
	var noImprovementCount int
	bestModel := *gpt

	fmt.Printf("Train (streaming): lines≈%d  Eval: from eval.en\n", linesCount)

	adamStep := 0
	workers := runtime.GOMAXPROCS(0)
	if s := os.Getenv("WORKERS"); s != "" {
		if v, err := strconv.Atoi(s); err == nil && v > 0 {
			workers = v
		}
	}
	if workers < 1 {
		workers = 1
	}

	for e := 0; e < params.Config.MaxEpochs; e++ {
		var totalLoss float64
		var steps float64
		var tokenCounter int
		var totalTokenLoss float64

		start := time.Now()

		// Build batch of random samples
		B := min(params.Config.BatchSize, 1000000000)
		samples := make([]sample, 0, B)
		for len(samples) < B {
			ids, err := iter.NextIDs()
			if err != nil {
				ids, err = iter.NextIDs()
				if err != nil {
					continue
				}
			}
			if len(ids) < 2 {
				continue
			}
			i := rand.Intn(len(ids) - 1)
			start := 0
			if i+1 > params.Config.SeqLen {
				start = i + 1 - params.Config.SeqLen
			}
			cp := make([]int, len(ids))
			copy(cp, ids)
			samples = append(samples, sample{ids: cp, i: i, start: start})
		}

		// Per-block accumulator struct
		type blkAcc struct {
			dWq, dWk, dWv []*mat.Dense
			dWo           *mat.Dense
			dWhid, dbHidden *mat.Dense
			dWout, dbOut    *mat.Dense
			dLn1G, dLn1B *mat.Dense
			dLn2G, dLn2B *mat.Dense
		}

		newAcc := func(g *transformer.Transformer) []blkAcc {
			acc := make([]blkAcc, params.Layers)
			for l := 0; l < params.Layers; l++ {
				att := g.Blocks[l].Attn
				mlp := g.Blocks[l].Mlp
				acc[l].dWq = make([]*mat.Dense, att.H)
				acc[l].dWk = make([]*mat.Dense, att.H)
				acc[l].dWv = make([]*mat.Dense, att.H)
				for h := 0; h < att.H; h++ {
					acc[l].dWq[h] = mat.NewDense(att.Wquery[h].RawMatrix().Rows, att.Wquery[h].RawMatrix().Cols, nil)
					acc[l].dWk[h] = mat.NewDense(att.Wkey[h].RawMatrix().Rows, att.Wkey[h].RawMatrix().Cols, nil)
					acc[l].dWv[h] = mat.NewDense(att.Wvalue[h].RawMatrix().Rows, att.Wvalue[h].RawMatrix().Cols, nil)
				}
				acc[l].dWo = mat.NewDense(att.Woutput.RawMatrix().Rows, att.Woutput.RawMatrix().Cols, nil)
				acc[l].dWhid = mat.NewDense(mlp.HiddenWeights.RawMatrix().Rows, mlp.HiddenWeights.RawMatrix().Cols, nil)
				acc[l].dbHidden = mat.NewDense(mlp.HiddenBias.RawMatrix().Rows, 1, nil)
				acc[l].dWout = mat.NewDense(mlp.OutputWeights.RawMatrix().Rows, mlp.OutputWeights.RawMatrix().Cols, nil)
				acc[l].dbOut = mat.NewDense(mlp.OutputBias.RawMatrix().Rows, 1, nil)
				acc[l].dLn1G = mat.NewDense(g.Blocks[l].Ln1.Gamma.RawMatrix().Rows, 1, nil)
				acc[l].dLn1B = mat.NewDense(g.Blocks[l].Ln1.Beta.RawMatrix().Rows, 1, nil)
				acc[l].dLn2G = mat.NewDense(g.Blocks[l].Ln2.Gamma.RawMatrix().Rows, 1, nil)
				acc[l].dLn2B = mat.NewDense(g.Blocks[l].Ln2.Beta.RawMatrix().Rows, 1, nil)
			}
			return acc
		}

		type workerOut struct {
			acc            []blkAcc
			dEmb, dPos     *mat.Dense
			totalLoss      float64
			steps          int
			totalTokenLoss float64
			tokenCounter   int
		}

		mergeAcc := func(dst, src []blkAcc) {
			for l := 0; l < len(dst); l++ {
				for h := 0; h < len(dst[l].dWq); h++ {
					dst[l].dWq[h].Add(dst[l].dWq[h], src[l].dWq[h])
					dst[l].dWk[h].Add(dst[l].dWk[h], src[l].dWk[h])
					dst[l].dWv[h].Add(dst[l].dWv[h], src[l].dWv[h])
				}
				dst[l].dWo.Add(dst[l].dWo, src[l].dWo)
				dst[l].dWhid.Add(dst[l].dWhid, src[l].dWhid)
				dst[l].dbHidden.Add(dst[l].dbHidden, src[l].dbHidden)
				dst[l].dWout.Add(dst[l].dWout, src[l].dWout)
				dst[l].dbOut.Add(dst[l].dbOut, src[l].dbOut)
				dst[l].dLn1G.Add(dst[l].dLn1G, src[l].dLn1G)
				dst[l].dLn1B.Add(dst[l].dLn1B, src[l].dLn1B)
				dst[l].dLn2G.Add(dst[l].dLn2G, src[l].dLn2G)
				dst[l].dLn2B.Add(dst[l].dLn2B, src[l].dLn2B)
			}
		}

		// Launch workers
		jobs := make(chan int, len(samples))
		results := make(chan workerOut, workers)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		for w := 0; w < workers; w++ {
			clone := gpt.CloneForGradsOnly()
			go func() {
				acc := newAcc(clone)
				dEmb := mat.NewDense(params.Emb.RawMatrix().Rows, params.Emb.RawMatrix().Cols, nil)
				dPos := mat.NewDense(params.PosEmb.RawMatrix().Rows, params.PosEmb.RawMatrix().Cols, nil)
				var tLoss float64
				var stps int
				var tokLoss float64
				var tokCnt int

				for {
					select {
					case idx, ok := <-jobs:
						if !ok {
							results <- workerOut{acc: acc, dEmb: dEmb, dPos: dPos, totalLoss: tLoss, steps: stps, totalTokenLoss: tokLoss, tokenCounter: tokCnt}
							return
						}
						sm := samples[idx]
						Xctx := IO.EmbedSequence(params.Emb, sm.ids[sm.start:sm.i+1])
						Y := Xctx
						for l := 0; l < params.Layers; l++ {
							Y = clone.Blocks[l].Forward(Y)
						}
						yLast := utils.LastCol(Y)
						logits := IO.Unembed(yLast)
						loss, _ := utils.CrossEntropyWithIndex(logits, sm.ids[sm.i+1])
						tLoss += loss
						stps++

						cols := Xctx.RawMatrix().Cols
						dY := dYbuf.Slice(0, params.Config.DModel, 0, cols).(*mat.Dense)
						// zero
						{
							r, c := dY.Dims()
							for ii := 0; ii < r; ii++ {
								for jj := 0; jj < c; jj++ {
									dY.Set(ii, jj, 0)
								}
							}
						}
						// output-path dEmb and dY
						dEmbOut := mat.NewDense(params.Emb.RawMatrix().Rows, params.Emb.RawMatrix().Cols, nil)
						seqTokLoss := 0.0
						for t := 0; t < cols-1; t++ {
							yCol := Y.Slice(0, params.Config.DModel, t, t+1).(*mat.Dense)
							logits := utils.ToDense(utils.Dot(params.Emb.T(), yCol))
							gold := sm.ids[t+1]
							ll, gLog := utils.CrossEntropyWithIndex(logits, gold)
							seqTokLoss += ll
							dyCol := utils.ToDense(utils.Dot(params.Emb, gLog))
							for ii := 0; ii < params.Config.DModel; ii++ {
								dY.Set(ii, t, dyCol.At(ii, 0))
							}
							dEmbOut.Add(dEmbOut, utils.ToDense(utils.Dot(yCol, gLog.T())))
						}
						tokLoss += seqTokLoss
						tokCnt += (cols - 1)

						// backprop stack (grads only)
						dX := dY
						for l := params.Layers - 1; l >= 0; l-- {
							dXl, dWq, dWk, dWv, dWo, dWhid, dbHidden, dWout, dbOut, dLn1G, dLn1B, dLn2G, dLn2B :=
								clone.Blocks[l].BackwardGradsOnlyFull(dX)
							for h := 0; h < len(dWq); h++ {
								acc[l].dWq[h].Add(acc[l].dWq[h], dWq[h])
								acc[l].dWk[h].Add(acc[l].dWk[h], dWk[h])
								acc[l].dWv[h].Add(acc[l].dWv[h], dWv[h])
							}
							acc[l].dWo.Add(acc[l].dWo, dWo)
							acc[l].dWhid.Add(acc[l].dWhid, dWhid)
							acc[l].dbHidden.Add(acc[l].dbHidden, dbHidden)
							acc[l].dWout.Add(acc[l].dWout, dWout)
							acc[l].dbOut.Add(acc[l].dbOut, dbOut)
							acc[l].dLn1G.Add(acc[l].dLn1G, dLn1G)
							acc[l].dLn1B.Add(acc[l].dLn1B, dLn1B)
							acc[l].dLn2G.Add(acc[l].dLn2G, dLn2G)
							acc[l].dLn2B.Add(acc[l].dLn2B, dLn2B)
							dX = dXl
						}
						// input path dEmbIn and dPos
						{
							dEmbIn := mat.NewDense(params.Emb.RawMatrix().Rows, params.Emb.RawMatrix().Cols, nil)
							for t := 0; t < cols; t++ {
								if sm.start+t >= len(sm.ids) {
									break
								}
								tokID := sm.ids[sm.start+t]
								colGrad := dX.Slice(0, params.Config.DModel, t, t+1).(*mat.Dense)
								for ii := 0; ii < params.Config.DModel; ii++ {
									dEmbIn.Set(ii, tokID, dEmbIn.At(ii, tokID)+colGrad.At(ii, 0))
								}
							}
							dEmbOut.Add(dEmbOut, dEmbIn)
							dEmb.Add(dEmb, dEmbOut)
						}
						{
							maxT := cols
							if maxT > params.PosEmb.RawMatrix().Cols {
								maxT = params.PosEmb.RawMatrix().Cols
							}
							for t := 0; t < maxT; t++ {
								for ii := 0; ii < params.Config.DModel; ii++ {
									dPos.Set(ii, t, dPos.At(ii, t)+dX.At(ii, t))
								}
							}
						}
					case <-ctx.Done():
						return
					}
				}
			}()
		}

		// feed jobs
		for i := range samples {
			jobs <- i
		}
		close(jobs)

		// collect
		accBatch := newAcc(gpt)
		dEmbBatch := mat.NewDense(params.Emb.RawMatrix().Rows, params.Emb.RawMatrix().Cols, nil)
		dPosBatch := mat.NewDense(params.PosEmb.RawMatrix().Rows, params.PosEmb.RawMatrix().Cols, nil)
		for w := 0; w < workers; w++ {
			out := <-results
			mergeAcc(accBatch, out.acc)
			dEmbBatch.Add(dEmbBatch, out.dEmb)
			dPosBatch.Add(dPosBatch, out.dPos)
			totalLoss += out.totalLoss
			steps += float64(out.steps)
			totalTokenLoss += out.totalTokenLoss
			tokenCounter += out.tokenCounter
		}

		// One optimizer step
		adamStep++
		attnLR := utils.LRSchedule(adamStep, params.Config.AttnLR)
		mlpLR := utils.LRSchedule(adamStep, params.Config.MLPLR)
		normLR := utils.LRSchedule(adamStep, params.Config.NormLR)
		unembedLR := utils.LRSchedule(adamStep, params.Config.UnembedLR)
		if adamStep < params.Config.WarmupSteps/2 {
			unembedLR = 0
		}
		for i := 0; i < params.Layers; i++ {
			gpt.Blocks[i].Attn.LearningRate = attnLR
			gpt.Blocks[i].Mlp.LearningRate = mlpLR
			gpt.Blocks[i].Ln1.LearningRate, gpt.Blocks[i].Ln2.LearningRate = normLR, normLR
		}

		// Average grads
		scale := 1.0 / float64(B)
		dEmbBatch.Scale(scale, dEmbBatch)
		dPosBatch.Scale(scale, dPosBatch)
		for l := 0; l < params.Layers; l++ {
			for h := 0; h < len(accBatch[l].dWq); h++ {
				accBatch[l].dWq[h].Scale(scale, accBatch[l].dWq[h])
				accBatch[l].dWk[h].Scale(scale, accBatch[l].dWk[h])
				accBatch[l].dWv[h].Scale(scale, accBatch[l].dWv[h])
			}
			accBatch[l].dWo.Scale(scale, accBatch[l].dWo)
			accBatch[l].dWhid.Scale(scale, accBatch[l].dWhid)
			accBatch[l].dbHidden.Scale(scale, accBatch[l].dbHidden)
			accBatch[l].dWout.Scale(scale, accBatch[l].dWout)
			accBatch[l].dbOut.Scale(scale, accBatch[l].dbOut)
			accBatch[l].dLn1G.Scale(scale, accBatch[l].dLn1G)
			accBatch[l].dLn1B.Scale(scale, accBatch[l].dLn1B)
			accBatch[l].dLn2G.Scale(scale, accBatch[l].dLn2G)
			accBatch[l].dLn2B.Scale(scale, accBatch[l].dLn2B)
		}

		// Clip and update
		if params.Config.GradClip > 0 {
			utils.ClipGrads(params.Config.GradClip, dEmbBatch, dPosBatch)
		}
		optimizations.InitEmbAdamIfNeeded()
		optimizations.InitPosAdamIfNeeded()
		params.EmbT++
		params.PosT++
		optimizations.AdamUpdateInPlace(params.Emb, dEmbBatch, params.EmbM, params.EmbV, params.EmbT,
			unembedLR, params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, params.Config.WeightDecay)
		optimizations.AdamUpdateInPlace(params.PosEmb, dPosBatch, params.PosM, params.PosV, params.PosT,
			params.Config.PosLR, params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, 0.0)

		// Per-block updates
		for l := 0; l < params.Layers; l++ {
			att := gpt.Blocks[l].Attn
			mlp := gpt.Blocks[l].Mlp
			ln1 := gpt.Blocks[l].Ln1
			ln2 := gpt.Blocks[l].Ln2
			att.T++; mlp.T++; ln1.T++; ln2.T++
			for h := 0; h < att.H; h++ {
				optimizations.AdamUpdateInPlace(att.Wquery[h], accBatch[l].dWq[h], att.MWq[h], att.VWq[h], att.T, attnLR,
					params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, params.Config.WeightDecay)
				optimizations.AdamUpdateInPlace(att.Wkey[h], accBatch[l].dWk[h], att.MWk[h], att.VWk[h], att.T, attnLR,
					params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, params.Config.WeightDecay)
				optimizations.AdamUpdateInPlace(att.Wvalue[h], accBatch[l].dWv[h], att.MWv[h], att.VWv[h], att.T, attnLR,
					params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, params.Config.WeightDecay)
			}
			optimizations.AdamUpdateInPlace(att.Woutput, accBatch[l].dWo, att.MWo, att.VWo, att.T, attnLR,
				params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, params.Config.WeightDecay)
			optimizations.AdamUpdateInPlace(mlp.HiddenWeights, accBatch[l].dWhid, mlp.MHiddenW, mlp.VHiddenW, mlp.T, mlpLR,
				params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, params.Config.WeightDecay)
			optimizations.AdamUpdateInPlace(mlp.HiddenBias, accBatch[l].dbHidden, mlp.MHiddenB, mlp.VHiddenB, mlp.T, mlpLR,
				params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, 0.0)
			optimizations.AdamUpdateInPlace(mlp.OutputWeights, accBatch[l].dWout, mlp.MOutputW, mlp.VOutputW, mlp.T, mlpLR,
				params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, params.Config.WeightDecay)
			optimizations.AdamUpdateInPlace(mlp.OutputBias, accBatch[l].dbOut, mlp.MOutputB, mlp.VOutputB, mlp.T, mlpLR,
				params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, 0.0)
			optimizations.AdamUpdateInPlace(ln1.Gamma, accBatch[l].dLn1G, ln1.MGamma, ln1.VGamma, ln1.T, normLR,
				params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, 0.0)
			optimizations.AdamUpdateInPlace(ln1.Beta, accBatch[l].dLn1B, ln1.MBeta, ln1.VBeta, ln1.T, normLR,
				params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, 0.0)
			optimizations.AdamUpdateInPlace(ln2.Gamma, accBatch[l].dLn2G, ln2.MGamma, ln2.VGamma, ln2.T, normLR,
				params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, 0.0)
			optimizations.AdamUpdateInPlace(ln2.Beta, accBatch[l].dLn2B, ln2.MBeta, ln2.VBeta, ln2.T, normLR,
				params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, 0.0)
		}

		// Epoch metrics
		avgTokLoss := 0.0
		trainPPL := 0.0
		if tokenCounter > 0 {
			avgTokLoss = totalTokenLoss / float64(tokenCounter)
			trainPPL = math.Exp(avgTokLoss)
		}
		corr, tot, ceSum := IO.EvaluateMetrics(*gpt)
		accuracy := 0.0
		evalPPL := 0.0
		if tot > 0 {
			accuracy = float64(corr) / float64(tot)
			evalPPL = math.Exp(ceSum / float64(tot))
		}

		 // cloze metrics
        cPrompts, cTok, cNLL := IO.EvaluateCloze(*gpt, 5000)
        clozePPL := 0.0
        if cTok > 0 {
            clozePPL = math.Exp(cNLL / float64(cTok))
        }
		fmt.Printf("Epoch %d - Acc: %.4f, TrainTokLoss: %.4f, TrainPPL: %.1f, EvalPPL: %.1f, ClozePPL: %.1f (n=%d) Steps: %f, Time: %v\n",
			e, accuracy, avgTokLoss, trainPPL, evalPPL, clozePPL, cPrompts, steps, time.Since(start))

		// Save checkpoints
		_ = safeSaveTransformer(gpt, "models/last_epoch.gob")
		if accuracy > bestAccuracy+params.Config.ImprovementThreshold && e > 20 {
			bestAccuracy = accuracy
			_ = transformer.SaveTransformer(gpt, "models/best_model.gob")
			bestModel = *gpt
			noImprovementCount = 0
		} else {
			noImprovementCount++
		}
		if (e+1)%params.Config.SaveEpochNumber == 0 {
			_ = safeSaveTransformer(gpt, fmt.Sprintf("models/epoch_%03d.gob", e+1))
			fmt.Printf("Saved checkpoint at epoch %d\n", e+1)
		}
		if noImprovementCount >= params.Config.Patience {
			fmt.Println("Stopping early due to no improvement.")
			break
		}
		if totalLoss/steps < params.Config.Epsilon {
			fmt.Println("Stopping early due to low loss.")
			break
		}
	}
	return &bestModel
}