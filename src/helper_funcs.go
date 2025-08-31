package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Guard functions
func chooseValidHeads(dModel, preferred int) int {
    if preferred <= 0 {
        return 1
    }
    if dModel%preferred == 0 { 
        return preferred
    }

    best := 1
    limit := preferred
    if limit > dModel {
        limit = dModel
    }
    for h := limit; h >= 1; h-- {
        if dModel%h == 0 {
            fmt.Printf("Warning: using %d heads instead of %d\n", h, preferred)
            best = h
            break
        }
    }
    return best
}

// Ensures grad has same T as forward pass
func expandGradToSeq(grad *mat.Dense, lastInput *mat.Dense) *mat.Dense {
	_, T := lastInput.Dims()
    gr, gc := grad.Dims()
    if gc == T {
        return grad
    }
    if gc == 1 && T > 1 {
        full := mat.NewDense(gr, T, nil)
        for i := 0; i < gr; i++ {
            full.Set(i, T-1, grad.At(i, 0))
        }
        return full
    }
    panic(fmt.Sprintf("expandGradToSeq: grad has %d cols, expected 1 or %d", gc, T))
}


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
	return mat.DenseCopyOf(m)
}
