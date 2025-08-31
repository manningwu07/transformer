package main

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func finiteDiffCheck(t *testing.T, name string, param *mat.Dense, grad *mat.Dense,
	forward func() float64, i, j int) {

	eps := 1e-5
	w0 := param.At(i, j)

	// Perturb +eps
	param.Set(i, j, w0+eps)
	lp := forward()

	// Perturb -eps
	param.Set(i, j, w0-eps)
	lm := forward()

	// Restore
	param.Set(i, j, w0)

	numGrad := (lp - lm) / (2.0 * eps)
	anaGrad := grad.At(i, j)

	if math.Abs(numGrad-anaGrad) > 1e-4 {
		t.Fatalf("%s[%d,%d] grad mismatch: num=%.6g ana=%.6g",
			name, i, j, numGrad, anaGrad)
	}
}

// ---- Attention ----
func TestAttentionGradCheck(t *testing.T) {
	rand.Seed(123)
	dModel := 4
	nHeads := 2
	attn := NewAttention(dModel, nHeads, 0.0)

	x := mat.NewDense(dModel, 3, randomArray(dModel*3, float64(dModel)))
	target := oneHot(dModel, 2)

	// Forward + loss
	forward := func() float64 {
		logits := attn.Forward(x)
		loss, _ := CrossEntropyWithGrad(lastCol(logits), target)
		return loss
	}

	// Analytic grads
	logits := attn.Forward(x)
	_, dL_dY := CrossEntropyWithGrad(lastCol(logits), target)
	_, dWq, dWk, dWv, dWo := attn.BackwardGradsOnly(dL_dY)

	// Check one element from each
	finiteDiffCheck(t, "Wquery", attn.Wquery[0], dWq[0], forward, 0, 0)
	finiteDiffCheck(t, "Wkey", attn.Wkey[0], dWk[0], forward, 0, 0)
	finiteDiffCheck(t, "Wvalue", attn.Wvalue[0], dWv[0], forward, 0, 0)
	finiteDiffCheck(t, "Woutput", attn.Woutput, dWo, forward, 0, 0)
}

// ---- MLP ----
func TestMLPGradCheck(t *testing.T) {
	rand.Seed(123)
	dModel := 4
	mlp := &MLP{
		inputs:        dModel,
		hiddens:       5,
		outputs:       dModel,
		learningRate:  0.0,
		hiddenWeights: mat.NewDense(5, dModel, randomArray(5*dModel, float64(dModel))),
		hiddenBias:    mat.NewDense(5, 1, nil),
		outputWeights: mat.NewDense(dModel, 5, randomArray(dModel*5, float64(dModel))),
		outputBias:    mat.NewDense(dModel, 1, nil),
	}

	x := mat.NewDense(dModel, 1, randomArray(dModel, float64(dModel)))
	target := oneHot(dModel, 2)

	forward := func() float64 {
		logits := mlp.Forward(x)
		loss, _ := CrossEntropyWithGrad(logits, target)
		return loss
	}

	logits := mlp.Forward(x)
	_, dL_dY := CrossEntropyWithGrad(logits, target)
	_, dWhid, _, dWout, _ := mlp.BackwardGradsOnly(dL_dY)

	// Check one element from each param
	finiteDiffCheck(t, "hiddenWeights", mlp.hiddenWeights, dWhid, forward, 0, 0)
	finiteDiffCheck(t, "outputWeights", mlp.outputWeights, dWout, forward, 0, 0)
}

// ---- Transformer Block ----
func TestBlockGradCheck(t *testing.T) {
	rand.Seed(123)
	dModel := 4
	block := TransformerBlock{
		attn: NewAttention(dModel, 2, 0.0),
		mlp: &MLP{
			inputs:        dModel,
			hiddens:       5,
			outputs:       dModel,
			learningRate:  0.0,
			hiddenWeights: mat.NewDense(5, dModel, randomArray(5*dModel, float64(dModel))),
			hiddenBias:    mat.NewDense(5, 1, nil),
			outputWeights: mat.NewDense(dModel, 5, randomArray(dModel*5, float64(dModel))),
			outputBias:    mat.NewDense(dModel, 1, nil),
		},
	}

	x := mat.NewDense(dModel, 3, randomArray(dModel*3, float64(dModel)))
	target := oneHot(dModel, 2)

	forward := func() float64 {
		logits := block.Forward(x)
		loss, _ := CrossEntropyWithGrad(lastCol(logits), target)
		return loss
	}

	logits := block.Forward(x)
	_, dL_dY := CrossEntropyWithGrad(lastCol(logits), target)
	_, dWq, _, _, _, dWhid, _ := block.BackwardGradsOnly(dL_dY)

	// Just check one param from attn and mlp
	finiteDiffCheck(t, "Block.Wquery", block.attn.Wquery[0], dWq[0], forward, 0, 0)
	finiteDiffCheck(t, "Block.hiddenWeights", block.mlp.hiddenWeights, dWhid, forward, 0, 0)
}

// ---- Full Transformer ----
func TestTransformerGradCheck(t *testing.T) {
	rand.Seed(123)
	layers = 2
	gpt := CreateGPT(4, 5, 4, 0.0, 0.0)

	x := mat.NewDense(4, 3, randomArray(12, 4))
	target := oneHot(4, 2)

	forward := func() float64 {
		Y := x
		for i := 0; i < layers; i++ {
			Y = gpt.blocks[i].Forward(Y)
		}
		loss, _ := CrossEntropyWithGrad(lastCol(Y), target)
		return loss
	}

	// Analytic grads via grads-only backprop across the stack
	Y := x
	for i := 0; i < layers; i++ {
		Y = gpt.blocks[i].Forward(Y)
	}
	_, dL_dY := CrossEntropyWithGrad(lastCol(Y), target)

	var dWq0 []*mat.Dense
	dY := dL_dY
	for i := layers - 1; i >= 0; i-- {
		dX, dWq, _, _, _, _, _ := gpt.blocks[i].BackwardGradsOnly(dY)
		if i == 0 {
			dWq0 = dWq
		}
		dY = dX
	}

	// Check one param from first block's attention
	finiteDiffCheck(t, "Transformer.Wquery",
		gpt.blocks[0].attn.Wquery[0], dWq0[0], forward, 0, 0)
}