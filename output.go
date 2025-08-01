package main

import (
	"bufio"
	"encoding/csv"
	"io"
	"math"
	"os"
	"strconv"
	"gonum.org/v1/gonum/mat"
)

func (net Network) CalculateLoss(inputData []float64, targetData []float64) float64 {
	// Forward propagation
	inputs := mat.NewDense(len(inputData), 1, inputData)
	hiddenInputs := add(dot(net.hiddenWeights, inputs), net.hiddenBias)
	hiddenOutputs := apply(sigmoid, hiddenInputs)
	finalInputs := add(dot(net.outputWeights, hiddenOutputs), net.outputBias)
	finalOutputs := apply(sigmoid, finalInputs)

	// Calculate loss
	targets := mat.NewDense(len(targetData), 1, targetData)
	outputErrors := subtract(targets, finalOutputs)
	
	// MSE = 1/N * sum(errors^2)
	sumOfSquares := 0.0
	r, _ := outputErrors.Dims()
	for i := 0; i < r; i++ {
		sumOfSquares += math.Pow(outputErrors.At(i, 0), 2)
	}
	return sumOfSquares / float64(r)
}

func evaluateAccuracy(net *Network) int {
	checkFile, _ := os.Open("mnist_dataset/mnist_test.csv")
	defer checkFile.Close()

	score := 0
	r := csv.NewReader(bufio.NewReader(checkFile))
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		inputs := make([]float64, net.inputs)
		for i := range inputs {
			if i == 0 {
				inputs[i] = 1.0
			}
			x, _ := strconv.ParseFloat(record[i], 64)
			inputs[i] = (x / 255.0 * 0.999) + 0.001
		}
		outputs := net.Predict(inputs)
		best := 0
		highest := 0.0
		for i := 0; i < net.outputs; i++ {
			if outputs.At(i, 0) > highest {
				best = i
				highest = outputs.At(i, 0)
			}
		}
		target, _ := strconv.Atoi(record[0])
		if best == target {
			score++
		}
	}

	return score
}