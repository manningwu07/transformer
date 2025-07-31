package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// asciiPlot draws a crude vertical bar chart of values (0..1).
func asciiPlot(values []float64) {
  const height = 10               // number of text rows
  n := len(values)
  if n == 0 {
    fmt.Println("no data to plot")
    return
  }
  // for each “row” from top (height) down to 1
  for row := height; row >= 1; row-- {
    threshold := float64(row) / float64(height)
    for _, v := range values {
      if v >= threshold {
        fmt.Print("█") // filled block
      } else {
        fmt.Print(" ")
      }
    }
    fmt.Println()
  }
  // x-axis
  fmt.Println(strings.Repeat("─", n))
  // optional: print epoch indices every 5 chars
  for i := range values {
    if i%5 == 0 {
      fmt.Print(strconv.Itoa(i % 10))
    } else {
      fmt.Print(" ")
    }
  }
  fmt.Println()
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