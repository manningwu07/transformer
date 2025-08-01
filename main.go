package main

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/csv"
	"flag"
	"fmt"
	"image"
	"image/png"
	"io"
	"math/rand"
	"os"
	"strconv"
	"time"
)

type TrainingRecord struct {
	inputs  []float64
	targets []float64
}

func main() {
	// 784 inputs - 28 x 28 pixels, each pixel is an input
	// 100 hidden nodes - an arbitrary number
	// 10 outputs - digits 0 to 9
	// 0.1 is the learning rate
	net := CreateNetwork(784, 200, 10, 0.110)

	mnist := flag.String("mnist", "", "Either train or predict to evaluate neural network")
	file := flag.String("file", "", "File name of 28 x 28 PNG file to evaluate")
	flag.Parse()

	// train or mass predict to determine the effectiveness of the trained network
	switch *mnist {
	case "train":
		mnistTrain(&net)
	case "predict":
		load(&net)
		mnistPredict(&net)
	default:
		// don't do anything
	}

	// predict individual digit images
	if *file != "" {
		// print the image out nicely on the terminal
		printImage(getImage(*file))
		// load the neural network from file
		load(&net)
		// predict which number it is
		fmt.Println("prediction:", predictFromImage(net, *file))
	}

}

// loadTrainingData reads the entire training dataset from the CSV file into memory.
func loadTrainingData(net *Network) ([]TrainingRecord, error) {
	fmt.Println("Loading training data...")
	trainFile, err := os.Open("mnist_dataset/mnist_train.csv")
	if err != nil {
		return nil, fmt.Errorf("error opening training file: %w", err)
	}
	defer trainFile.Close()

	r := csv.NewReader(bufio.NewReader(trainFile))
	var trainingData []TrainingRecord

	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error reading CSV record: %w", err)
		}

		inputs := make([]float64, net.inputs)
		for i := 0; i < net.inputs; i++ {
			x, _ := strconv.ParseFloat(record[i], 64)
			inputs[i] = (x/255.0*0.999) + 0.001
		}

		targets := make([]float64, 10)
		for i := range targets {
			targets[i] = 0.001
		}
		x, _ := strconv.Atoi(record[0])
		targets[x] = 0.999

		trainingData = append(trainingData, TrainingRecord{inputs, targets})
	}

	return trainingData, nil
}

func mnistTrain(net *Network) {
	rand.Seed(time.Now().UTC().UnixNano())
	t1 := time.Now()

	// --- Mini-Batch SGD and Early Stopping Variables ---
	// Load all training data into memory once to avoid slow file I/O in each epoch.
	trainingData, err := loadTrainingData(net)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("Finished loading %d training records.\n", len(trainingData))

	// Early stopping parameters
	const upperBound = 500
	const patience = 20
	const epsilon = 1e-6
	const batchSize = 512

	var bestAccuracy float64 = 0.0
	var noImprovementCount int
	var bestModel Network
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
	for e := 0; e < upperBound; e++ {
		var totalLoss float64
		var totalRecords float64

		// Randomly shuffle the training data at the beginning of each epoch.
		// This is a key step for mini-batch SGD.
		rand.Shuffle(len(trainingData), func(i, j int) {
			trainingData[i], trainingData[j] = trainingData[j], trainingData[i]
		})

		// Use a mini-batch of 1024 of the shuffled data for training in this epoch.
		miniBatch := trainingData[:batchSize]

		// Iterate through the mini-batch
		for _, record := range miniBatch {
			net.Train(record.inputs, record.targets)
			totalLoss += net.CalculateLoss(record.inputs, record.targets)
			totalRecords++
		}
		
		// Calculate average loss for the epoch
		avgLoss := totalLoss / totalRecords

		// Evaluate accuracy on the test set
		correct := evaluateAccuracy(net)
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
		if accuracy > bestAccuracy {
			bestAccuracy = accuracy
			// Save the model if we find a better one.
			bestModel = *net // Create a deep copy of the current network.
			noImprovementCount = 0
		} else {
			// If there is no improvement, increment the counter.
			noImprovementCount++
		}

		// The loop now breaks if we've seen enough epochs without a new best accuracy.
		if noImprovementCount >= patience {
			fmt.Println("\nStopping training early due to lack of improvement in accuracy.")
			break
		}
		// If the loss func is too small, stop training.
		if avgLoss < epsilon {
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

func mnistPredict(net *Network) {
	t1 := time.Now()
	score := evaluateAccuracy(net)
	elapsed := time.Since(t1)
	fmt.Printf("Time taken to check: %s\n", elapsed)
	fmt.Println("score:", score)
}

// print out image on iTerm2; equivalent to imgcat on iTerm2
func printImage(img image.Image) {
	var buf bytes.Buffer
	png.Encode(&buf, img)
	imgBase64Str := base64.StdEncoding.EncodeToString(buf.Bytes())
	fmt.Printf("\x1b]1337;File=inline=1:%s\a\n", imgBase64Str)
}

// get the file as an image
func getImage(filePath string) image.Image {
	imgFile, err := os.Open(filePath)
	defer imgFile.Close()
	if err != nil {
		fmt.Println("Cannot read file:", err)
	}
	img, _, err := image.Decode(imgFile)
	if err != nil {
		fmt.Println("Cannot decode file:", err)
	}
	return img
}
