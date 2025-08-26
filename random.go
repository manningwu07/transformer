package main

import (
	"math"
	"math/rand"
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