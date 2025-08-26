package main

import (
	"encoding/csv"
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)

// TrainingRecord is a generic dataset example (inputs column vector,
// targets column vector, both mat.Dense wrappers are easy to create).
type TrainingRecord struct {
	Inputs  *mat.Dense
	Targets *mat.Dense
}

// DummyDataset creates N synthetic training records for testing.
// Each input and target are column vectors of length dim.
// Targets are one-hot encoded for a random class. This is useful for
// exercising forward/backward and verifying loss decreases.
func DummyDataset(n, dim int) []TrainingRecord {
	out := make([]TrainingRecord, n)
	for i := 0; i < n; i++ {
		// simple deterministic synthetic inputs (small values)
		values := make([]float64, dim)
		for j := 0; j < dim; j++ {
			values[j] = float64((i+j)%7) * 0.01 // deterministic small pattern
		}
		inputVec := mat.NewDense(dim, 1, values)

		targetIdx := (i % dim)
		targetVec := make([]float64, dim)
		for k := 0; k < dim; k++ {
			if k == targetIdx {
				targetVec[k] = 1.0
			} else {
				targetVec[k] = 0.0
			}
		}
		targetDense := mat.NewDense(dim, 1, targetVec)

		out[i] = TrainingRecord{
			Inputs:  inputVec,
			Targets: targetDense,
		}
	}
	return out
}

// LoadCSVAsDataset loads a CSV where each row is: t0,t1,...,t{d-1},label
// It returns TrainingRecord with vector inputs (d x 1) and one-hot target
// length = numClasses. labelColumn is the index of the label in the CSV
// row (-1 if it is the first column as in MNIST).
func LoadCSVAsDataset(path string, dim, numClasses, labelColumn int) ([]TrainingRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	records := []TrainingRecord{}
	for {
		row, err := r.Read()
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return nil, err
		}
		if len(row) < dim+1 {
			return nil, fmt.Errorf("row length %d < expected %d", len(row), dim+1)
		}

		inputs := make([]float64, dim)
		ci := 0
		for j := 0; j < len(row); j++ {
			if j == labelColumn {
				continue
			}
			// parse float
			var v float64
			fmt.Sscan(row[j], &v)
			inputs[ci] = v
			ci++
			if ci >= dim {
				break
			}
		}

		var label int
		if labelColumn >= 0 {
			fmt.Sscan(row[labelColumn], &label)
		} else {
			// assume label is first column
			fmt.Sscan(row[0], &label)
		}
		targets := make([]float64, numClasses)
		for k := 0; k < numClasses; k++ {
			if k == label {
				targets[k] = 1.0
			} else {
				targets[k] = 0.0
			}
		}

		records = append(records, TrainingRecord{
			Inputs:  mat.NewDense(dim, 1, inputs),
			Targets: mat.NewDense(numClasses, 1, targets),
		})
	}
	return records, nil
}