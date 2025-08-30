package main

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// randomArray returns 'size' samples from U(-1/sqrt(v), 1/sqrt(v)).
// It uses the global math/rand RNG (seed in main for determinism).
func randomArray(size int, v float64) []float64 {
	min := -1.0 / math.Sqrt(v+1e-12)
	max := 1.0 / math.Sqrt(v+1e-12)
	out := make([]float64, size)
	for i := 0; i < size; i++ {
		out[i] = min + (max-min)*rand.Float64()
	}
	return out
}

// Helper functions

func oneHot(n, idx int) *mat.Dense {
	v := make([]float64, n)
	if idx >= 0 && idx < n {
		v[idx] = 1.0
	}
	return mat.NewDense(n, 1, v)
}

func toDense(m mat.Matrix) *mat.Dense {
	if d, ok := m.(*mat.Dense); ok {
		return d
	}
	return mat.DenseCopyOf(m) // safely materialize
}
