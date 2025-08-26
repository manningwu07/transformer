package main

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// Finite-difference check for dL/dWquery[i,j]
func TestAttentionWqGradFiniteDiff(t *testing.T) {
	rand.Seed(123)
	layers = 1 // isolate single attention block

	d := 4
	attn := Attention{
		Wquery:  mat.NewDense(d, d, randomArray(d*d, float64(d))),
		Wkey:    mat.NewDense(d, d, randomArray(d*d, float64(d))),
		Wvalue:  mat.NewDense(d, d, randomArray(d*d, float64(d))),
		Woutput: mat.NewDense(d, d, randomArray(d*d, float64(d))),
		learningRate:      0.0, // no updates during test
	}

	x := mat.NewDense(d, 1, []float64{0.05, -0.02, 0.03, 0.01})
	target := oneHot(d, 2)

	// Forward
	logits := attn.Forward(x)
	_, dL_dY := CrossEntropyWithGrad(logits, target)

	// Analytic grads
	_, dWq, _, _, _ := attn.BackwardGradsOnly(dL_dY)

	// Finite difference for one element
	i, j := 1, 2
	eps := 1e-5
	w0 := attn.Wquery.At(i, j)

	attn.Wquery.Set(i, j, w0+eps)
	lp, _ := CrossEntropyWithGrad(attn.Forward(x), target)

	attn.Wquery.Set(i, j, w0-eps)
	lm, _ := CrossEntropyWithGrad(attn.Forward(x), target)

	attn.Wquery.Set(i, j, w0) // restore

	numGrad := (lp - lm) / (2.0 * eps)
	anaGrad := dWq.At(i, j)

	if math.Abs(numGrad-anaGrad) > 1e-4 {
		t.Fatalf("Wquery[%d,%d] grad mismatch: num=%.6g ana=%.6g",
			i, j, numGrad, anaGrad)
	}
}