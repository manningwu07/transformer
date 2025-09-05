package main

import (
	"math"
	"math/rand"
	"testing"

	"github.com/manningwu07/GPT/params"
	"github.com/manningwu07/GPT/transformer"
	"github.com/manningwu07/GPT/utils"
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
	attn := transformer.NewAttention(dModel, nHeads, 0.0)

	x := mat.NewDense(dModel, 3, utils.RandomArray(dModel*3, float64(dModel)))

	// Forward + loss
	forward := func() float64 {
		logits := attn.Forward(x)
		loss, _ := utils.CrossEntropyWithIndex(utils.LastCol(logits), 2)
		return loss
	}

	// Analytic grads
	logits := attn.Forward(x)
	_, dL_dY := utils.CrossEntropyWithIndex(utils.LastCol(logits), 2)
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
	mlp := &transformer.MLP{
		Inputs:        dModel,
		Hiddens:       5,
		Outputs:       dModel,
		LearningRate:  0.0,
		HiddenWeights: mat.NewDense(5, dModel, utils.RandomArray(5*dModel, float64(dModel))),
		HiddenBias:    mat.NewDense(5, 1, nil),
		OutputWeights: mat.NewDense(dModel, 5, utils.RandomArray(dModel*5, float64(dModel))),
		OutputBias:    mat.NewDense(dModel, 1, nil),
	}

	x := mat.NewDense(dModel, 1, utils.RandomArray(dModel, float64(dModel)))

	forward := func() float64 {
		logits := mlp.Forward(x)
		loss, _ := utils.CrossEntropyWithIndex(logits, 2)
		return loss
	}

	logits := mlp.Forward(x)
	_, dL_dY := utils.CrossEntropyWithIndex(logits, 2)
	_, dWhid, _, dWout, _ := mlp.BackwardGradsOnly(dL_dY)

	// Check one element from each param
	finiteDiffCheck(t, "hiddenWeights", mlp.HiddenWeights, dWhid, forward, 0, 0)
	finiteDiffCheck(t, "outputWeights", mlp.OutputWeights, dWout, forward, 0, 0)
}

// ---- Transformer Block ----
func TestBlockGradCheck(t *testing.T) {
	rand.Seed(123)
	dModel := 4
	block := transformer.TransformerBlock{
		Attn: transformer.NewAttention(dModel, 2, 0.0),
		Mlp: &transformer.MLP{
			Inputs:        dModel,
			Hiddens:       5,
			Outputs:       dModel,
			LearningRate:  0.0,
			HiddenWeights: mat.NewDense(5, dModel, utils.RandomArray(5*dModel, float64(dModel))),
			HiddenBias:    mat.NewDense(5, 1, nil),
			OutputWeights: mat.NewDense(dModel, 5, utils.RandomArray(dModel*5, float64(dModel))),
			OutputBias:    mat.NewDense(dModel, 1, nil),
		},
	}

	x := mat.NewDense(dModel, 3, utils.RandomArray(dModel*3, float64(dModel)))

	forward := func() float64 {
		logits := block.Forward(x)
		loss, _ := utils.CrossEntropyWithIndex(utils.LastCol(logits), 2)
		return loss
	}

	logits := block.Forward(x)
	_, dL_dY := utils.CrossEntropyWithIndex(utils.LastCol(logits), 2)
	_, dWq, _, _, _, dWhid, _ := block.BackwardGradsOnly(dL_dY)

	// Just check one param from attn and mlp
	finiteDiffCheck(t, "Block.Wquery", block.Attn.Wquery[0], dWq[0], forward, 0, 0)
	finiteDiffCheck(t, "Block.hiddenWeights", block.Mlp.HiddenWeights, dWhid, forward, 0, 0)
}

// ---- Full Transformer ----
func TestTransformerGradCheck(t *testing.T) {
	rand.Seed(123)
	params.Layers = 2
	gpt := transformer.CreateGPT(4, 5, 4, 0.0, 0.0)

	x := mat.NewDense(4, 3, utils.RandomArray(12, 4))

	forward := func() float64 {
		Y := x
		for i := 0; i < params.Layers; i++ {
			Y = gpt.Blocks[i].Forward(Y)
		}
		loss, _ := utils.CrossEntropyWithIndex(utils.LastCol(Y), 2)
		return loss
	}

	// Analytic grads via grads-only backprop across the stack
	Y := x
	for i := 0; i < params.Layers; i++ {
		Y = gpt.Blocks[i].Forward(Y)
	}
	_, dL_dY := utils.CrossEntropyWithIndex(utils.LastCol(Y), 2)

	var dWq0 []*mat.Dense
	dY := dL_dY
	for i := params.Layers - 1; i >= 0; i-- {
		dX, dWq, _, _, _, _, _ := gpt.Blocks[i].BackwardGradsOnly(dY)
		if i == 0 {
			dWq0 = dWq
		}
		dY = dX
	}

	// Check one param from first block's attention
	finiteDiffCheck(t, "Transformer.Wquery",
		gpt.Blocks[0].Attn.Wquery[0], dWq0[0], forward, 0, 0)
}