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

func vector(vals []float64) *mat.Dense {
	return mat.NewDense(len(vals), 1, vals)
}

func headVec(v *mat.Dense, k int) []float64 {
	r, c := v.Dims()
	if c != 1 {
		return []float64{}
	}
	if k > r {
		k = r
	}
	out := make([]float64, k)
	for i := 0; i < k; i++ {
		out[i] = v.At(i, 0)
	}
	return out
}

func rowSums(m *mat.Dense) []float64 {
	r, c := m.Dims()
	out := make([]float64, r)
	for i := 0; i < r; i++ {
		sum := 0.0
		for j := 0; j < c; j++ {
			sum += m.At(i, j)
		}
		out[i] = sum
	}
	return out
}

func forwardThrough(gpt Transformer, x *mat.Dense) *mat.Dense {
	out := x
	for i := 0; i < layers; i++ {
		out = gpt.blocks[i].Forward(out)
	}
	return out
}