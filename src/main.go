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
    DModel     int     // embedding / model dimension
    HiddenSize int     // hidden size for MLP
    NumClasses int     // output size (usually same as vocab size)
    AttnLR     float64 // learning rate for attention
    MLPLR      float64 // learning rate for MLP
	UnembedLR  float64 // learning rate for unembedding (embZH)

    MaxEpochs  int     // maximum number of epochs
    Patience   int     // early stopping patience
    Epsilon    float64 // stop if loss < epsilon
    BatchSize  int     // mini-batch size
	ValFrac    float64 // fraction of data held out for validation
}

// Reasonable defaults for small experiments
var config = TrainingConfig{
    DModel:     64,     // small but not too tiny
    HiddenSize: 64,     // bigger hidden layer for MLP
    NumClasses: 64,   // match vocab size (same as DModel for now)
    AttnLR:     0.02,   // smaller LR than 0.10 for stability
    MLPLR:      0.02,   // same for MLP
	UnembedLR:  0.02,

    MaxEpochs:  100,    // not 500 — faster experiments
    Patience:   10,     // stop earlier if no improvement
    Epsilon:    1e-4,   // don’t wait for 1e-6, too strict
    BatchSize:  4096,    // smaller batch for toy data
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
        config.NumClasses,
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
        valN = min(1000, len(trainingData)/10)
        if valN < 1 {
            valN = 1
        }
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
		end := config.BatchSize
		if end > len(trainingData) {
			end = len(trainingData)
		}
		miniBatch := trainingData[:end]

		// Iterate through the mini-batch
		for _, record := range miniBatch {
			// Forward pass
			X := record.Inputs
			for i := 0; i < layers; i++ {
				X = gpt.blocks[i].Forward(X)
			}

			logits := UnembedZH(X)

			// Loss + gradient
			target := record.Targets
			loss, gradLogits := CrossEntropyWithGrad(logits, target)

			totalLoss += loss
			totalRecords++

			// Backprop through unembedding: logits = embZH^T * X
            // dX = embZH * (p - t)
            dX := dot(embZH, gradLogits).(*mat.Dense)

            // Optionally train the unembedding (recommended)
            // dEmbZH = X * (p - t)^T
            dEmb := dot(X, gradLogits.T()).(*mat.Dense)
            embZH = add(embZH, scale(-config.UnembedLR, dEmb)).(*mat.Dense)

			// Backward pass (reverse order)
			for i := layers - 1; i >= 0; i-- {
				dX = gpt.blocks[i].Backward(dX)
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