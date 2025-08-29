package main

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// Finite-difference check for dL/dWquery[h][i,j] in multi-head attention
func TestMHAWqGradFiniteDiff(t *testing.T) {
	rand.Seed(123)
	layers = 1 // isolate single attention block

	dModel := 4
	nHeads := 2
	attn := NewAttention(dModel, nHeads, 0.0) // lr=0.0 so no updates

	// Simple input and target
	x := mat.NewDense(dModel, 1, []float64{0.05, -0.02, 0.03, 0.01})
	target := oneHot(dModel, 2)

	// Forward
	logits := attn.Forward(x)
	_, dL_dY := CrossEntropyWithGrad(logits, target)

	// Analytic grads
	_, dWq, _, _, _ := attn.BackwardGradsOnly(dL_dY)

	// Pick a head and element to test
	h := 0
	i, j := 1, 2
	eps := 1e-5
	w0 := attn.Wquery[h].At(i, j)

	// Perturb +eps
	attn.Wquery[h].Set(i, j, w0+eps)
	lp, _ := CrossEntropyWithGrad(attn.Forward(x), target)

	// Perturb -eps
	attn.Wquery[h].Set(i, j, w0-eps)
	lm, _ := CrossEntropyWithGrad(attn.Forward(x), target)

	// Restore
	attn.Wquery[h].Set(i, j, w0)

	// Numerical vs analytic
	numGrad := (lp - lm) / (2.0 * eps)
	anaGrad := dWq[h].At(i, j)

	if math.Abs(numGrad-anaGrad) > 1e-4 {
		t.Fatalf("Head %d Wquery[%d,%d] grad mismatch: num=%.6g ana=%.6g",
			h, i, j, numGrad, anaGrad)
	}
}