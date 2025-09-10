package utils

import (
	"fmt"
	"math"
	"math/rand"
	"gonum.org/v1/gonum/mat"
)

// Guard functions
func ChooseValidHeads(dModel, preferred int) int {
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
func ExpandGradToSeq(grad *mat.Dense, lastInput *mat.Dense) *mat.Dense {
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

func RandomArray(size int, v float64) []float64 {
	min := -1.0 / math.Sqrt(v+1e-12)
	max := 1.0 / math.Sqrt(v+1e-12)
	out := make([]float64, size)
	for i := 0; i < size; i++ {
		out[i] = min + (max-min)*rand.Float64()
	}
	return out
}

// Helper functions

func OneHot(n, idx int) *mat.Dense {
	v := make([]float64, n)
	if idx >= 0 && idx < n {
		v[idx] = 1.0
	}
	return mat.NewDense(n, 1, v)
}

func ToDense(m mat.Matrix) *mat.Dense {
	if d, ok := m.(*mat.Dense); ok {
		return d
	}
	return mat.DenseCopyOf(m)
}

func MatrixNorm(m *mat.Dense) float64 {
	r, c := m.Dims()
	s := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := m.At(i, j)
			s += v * v
		}
	}
	return math.Sqrt(s)
}

// debugging and clipping.

// clipGrads scales all grads so their combined norm <= maxNorm.
// Returns the scale actually applied (<=1.0) or 1.0 if no clip.
func ClipGrads(maxNorm float64, grads ...*mat.Dense) float64 {
	if maxNorm <= 0 {
		return 1.0
	}
	sum := 0.0
	for _, g := range grads {
		if g == nil {
			continue
		}
		n := matFroNorm(g)
		sum += n * n
	}
	gn := math.Sqrt(sum)
	if gn <= maxNorm || gn == 0 {
		return 1.0
	}
	s := maxNorm / gn
	for _, g := range grads {
		if g != nil {
			scaleInPlace(g, s)
		}
	}
	return s
}

func matFroNorm(a *mat.Dense) float64 {
	r, c := a.Dims()
	s := 0.0
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := a.At(i, j)
			s += v * v
		}
	}
	return math.Sqrt(s)
}

func scaleInPlace(a *mat.Dense, s float64) {
	if s == 1.0 {
		return
	}
	r, c := a.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			a.Set(i, j, a.At(i, j)*s)
		}
	}
}

func OnesLike(a *mat.Dense) *mat.Dense {
	r, c := a.Dims()
	out := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out.Set(i, j, 1)
		}
	}
	return out
}