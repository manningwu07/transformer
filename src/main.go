package main

import (
	"encoding/csv"
	"os"
	"strconv"
	"time"

	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Keep transformer.go unchanged. We define layers here
// because transformer.go references it.
var layers = 2

// TrainingConfig holds all tunable hyperparameters in one place.
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
	DModel:     512,   // try 512 or 768; go 1024 if you can
	HiddenSize: 2048,  // ~4x dModel
	VocabSize:  8192,  // top 1–4 char pieces
	NumHeads:   8,     // dHead = DModel/NumHeads
	SeqLen:     128,   // max context
	AttnLR:     0.003, // simple SGD -> smaller LRs
	MLPLR:      0.003,
	UnembedLR:  0.003,

	MaxEpochs:  50,
	Patience:   10,
	Epsilon:    1e-4,
	BatchSize:  1024, // each example is one prefix
	ValFrac:    0.1,
}

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()

	// --- Mini-Batch SGD and Early Stopping Variables ---
	// Load all training data into memory once to avoid slow file I/O in each epoch.

	gpt := CreateGPT(
		config.DModel,
		config.HiddenSize,
		config.VocabSize,
		config.AttnLR,
		config.MLPLR,
	)
	trainingData, err := loadTrainingSet(gpt)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Finished loading %d training records.\n", len(trainingData))

	var bestAccuracy float64 = -1.0
	var noImprovementCount int
	bestModel := gpt

	rand.Shuffle(len(trainingData), func(i, j int) {
		trainingData[i], trainingData[j] = trainingData[j], trainingData[i]
	})
	valN := int(config.ValFrac * float64(len(trainingData)))
	if valN < 1 {
		valN = max(min(1000, len(trainingData)/10), 1)
	}
	valRecords := trainingData[:valN]
	trainRecords := trainingData[valN:]
	fmt.Printf("Train: %d  Val: %d\n", len(trainRecords), len(valRecords))

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
		var totalRecords float64

		// Randomly shuffle the training data at the beginning of each epoch.
		rand.Shuffle(len(trainingData), func(i, j int) {
			trainingData[i], trainingData[j] = trainingData[j], trainingData[i]
		})

		// Use a mini-batch for training in this epoch.
		end := min(config.BatchSize, len(trainingData))
		miniBatch := trainingData[:end]

		// Iterate through the mini-batch
		for _, record := range miniBatch {
			// Forward pass on context matrix (d x T)
			Xctx := record.Inputs
			Y := Xctx
			for i := 0; i < layers; i++ {
				Y = gpt.blocks[i].Forward(Y)
			}
			// take last position only
			yLast := lastCol(Y)
			logits := Unembed(yLast)

			// Loss + gradient
			target := record.Targets
			loss, gradLogits := CrossEntropyWithGrad(logits, target)

			totalLoss += loss
			totalRecords++

			// Backprop through tied unembedding: logits = emb^T * yLast
			// dyLast = emb * (p - t)
			dyLast := dot(emb, gradLogits).(*mat.Dense) // (d x 1)
			// dEmb = yLast * (p - t)^T
			dEmb := dot(yLast, gradLogits.T()).(*mat.Dense)
			emb = add(emb, scale(-config.UnembedLR, dEmb)).(*mat.Dense)

			dY := mat.NewDense(config.DModel, Xctx.RawMatrix().Cols, nil)
			for i := 0; i < config.DModel; i++ {
				dY.Set(i, dY.RawMatrix().Cols-1, dyLast.At(i, 0))
			}
			for i := layers - 1; i >= 0; i-- {
				dY = gpt.blocks[i].Backward(dY)
			}
		}

		// Calculate average loss for the epoch
		avgLoss := totalLoss / totalRecords

		// Evaluate accuracy on the test set
		correct, total := evaluateAccuracy(gpt)
		var accuracy float64
		if total > 0 {
			accuracy = float64(correct) / float64(total)
		} else {
			accuracy = 0
		}

		// Log the epoch's metrics to the CSV file
		logWriter.Write([]string{
			strconv.Itoa(e + 1),
			strconv.FormatFloat(accuracy, 'f', 4, 64),
			strconv.FormatFloat(avgLoss, 'f', 4, 64),
		})

		fmt.Printf("Epoch %d - Accuracy: %.4f, Loss: %.4f\n", e+1, accuracy, avgLoss)

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
