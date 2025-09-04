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

	fmt.Printf("Train (streaming): lines≈%d  Eval: from eval.en\n", linesCount)

	bestModel := TrainGPT(&gpt, iter, linesCount)

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