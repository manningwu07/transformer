package main

import (
	"encoding/csv"
	"os"
	"strconv"
	"time"

	"fmt"
	"math/rand"
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

    MaxEpochs  int     // maximum number of epochs
    Patience   int     // early stopping patience
    Epsilon    float64 // stop if loss < epsilon
    BatchSize  int     // mini-batch size
}

// Reasonable defaults for small experiments
var config = TrainingConfig{
    DModel:     16,     // small but not too tiny
    HiddenSize: 32,     // bigger hidden layer for MLP
    NumClasses: 16,     // match vocab size (same as DModel for now)
    AttnLR:     0.05,   // smaller LR than 0.10 for stability
    MLPLR:      0.05,   // same for MLP

    MaxEpochs:  100,    // not 500 — faster experiments
    Patience:   10,     // stop earlier if no improvement
    Epsilon:    1e-4,   // don’t wait for 1e-6, too strict
    BatchSize:  128,    // smaller batch for toy data
}

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()

	// --- Mini-Batch SGD and Early Stopping Variables ---
	// Load all training data into memory once to avoid slow file I/O in each epoch.

	gpt := CreateGPT(config.DModel, config.HiddenSize, config.NumClasses, config.AttnLR, config.MLPLR)
	trainingData, err := loadTrainingSet(gpt)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Finished loading %d training records.\n", len(trainingData))

	var bestAccuracy float64 = 0.0
	var noImprovementCount int
	var bestModel Transformer
	// --- End Mini-Batch SGD and Early Stopping Variables ---

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

	// The main training loop now uses a dynamic condition instead of a fixed epoch count.
	for e := 0; e < config.MaxEpochs; e++ {
		var totalLoss float64
		var totalRecords float64

		// Randomly shuffle the training data at the beginning of each epoch.
		// This is a key step for mini-batch SGD.
		rand.Shuffle(len(trainingData), func(i, j int) {
			trainingData[i], trainingData[j] = trainingData[j], trainingData[i]
		})

		// Use a mini-batch of 1024 of the shuffled data for training in this epoch.
		miniBatch := trainingData[:config.BatchSize]

		// Iterate through the mini-batch
		for _, record := range miniBatch {
			// Forward pass
			X := record.Inputs
			for i := 0; i < layers; i++ {
				X = gpt.blocks[i].Forward(X)
			}

			// Loss + gradient
			target := record.Targets
			loss, grad := CrossEntropyWithGrad(X, target)

			totalLoss += loss
			totalRecords++

			// Backward pass (reverse order)
			for i := layers - 1; i >= 0; i-- {
				grad = gpt.blocks[i].Backward(grad)
			}
		}

		// Calculate average loss for the epoch
		avgLoss := totalLoss / totalRecords

		// Evaluate accuracy on the test set
		correct := evaluateAccuracy(gpt)
		accuracy := float64(correct) / 10000.0

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
	save(bestModel)
	fmt.Println("Saved the best performing model.")
}

// func main() {
// 	// Deterministic
// 	rand.Seed(42)

// 	// Square model so residual adds are valid
// 	d := 8
// 	attnLR := 0.10
// 	mlpLR := 0.10
// 	gpt := CreateGPT(d, d, d, attnLR, mlpLR)

// 	x := vector([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8})
// 	target := oneHot(d, 3)

// 	// First forward
// 	logits := forwardThrough(gpt, x)
// 	loss0, grad := CrossEntropyWithGrad(logits, target)
// 	fmt.Printf("Initial loss: %.6f\n", loss0)

// 	// Inspect attention weights of block 0 (row sums should be ~1)
// 	for i := 0; i < d; i++ {
// 		if A := gpt.blocks[0].attn.A; A != nil {
// 			fmt.Printf("Block 0 attention row sums: %v\n", rowSums(A[i]))
// 		}
// 	}

// 	// One backward step through all layers
// 	for i := layers - 1; i >= 0; i-- {
// 		grad = gpt.blocks[i].Backward(grad)
// 	}

// 	// Forward again
// 	logits2 := forwardThrough(gpt, x)
// 	loss1, _ := CrossEntropyWithGrad(logits2, target)
// 	fmt.Printf("Loss after 1 step: %.6f\n", loss1)
// 	fmt.Printf("Loss decreased: %v\n", loss1 < loss0)
// 	fmt.Printf("Logits (first run) head: %v\n", headVec(logits, 5))
// 	fmt.Printf("Logits (after 1 step) head: %v\n", headVec(logits2, 5))
// }
