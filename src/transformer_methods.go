package main

import (
	"os"

	"gonum.org/v1/gonum/mat"
)

type Transformer struct {
	blocks []TransformerBlock
}

type TransformerBlock struct {
	attn *Attention
	mlp  *MLP
}

type Attention struct {
	H            int
	dModel       int
	dHead        int
	Wquery       []*mat.Dense
	Wkey         []*mat.Dense
	Wvalue       []*mat.Dense
	Woutput      *mat.Dense
	learningRate float64

	// cache for backprop
	X       *mat.Dense
	Q, K, V []*mat.Dense
	Scores  []*mat.Dense
	A       []*mat.Dense
	O       []*mat.Dense
	O_cat   *mat.Dense

	// Performance optimization
	maskCache map[int]*mat.Dense
	lastT     int
	parallel  bool // parallelize over heads if true
}

type MLP struct {
	inputs, hiddens, outputs  int
	hiddenWeights, hiddenBias *mat.Dense
	outputWeights, outputBias *mat.Dense
	learningRate              float64

	// cache for backprop
	lastInput, hiddenOutputs, finalOutputs *mat.Dense
}

// Initalization

func CreateGPT(dModel, hidden, vocabSize int, AttnRate float64, MLPRate float64) Transformer {
	gpt := Transformer{
		blocks: make([]TransformerBlock, layers),
	}

    numHeads := chooseValidHeads(dModel, config.NumHeads)

	for i := range layers {
		attn := NewAttention(dModel, numHeads, AttnRate)

		// MLP IS WAYYY TOO BIG, PLEASE FIX THIS
		mlp := &MLP{
			inputs:        dModel,
			hiddens:       hidden,
			outputs:       dModel,
			learningRate:  MLPRate,
			hiddenWeights: mat.NewDense(hidden, dModel, randomArray(dModel*hidden, float64(dModel))),
			hiddenBias:    mat.NewDense(hidden, 1, nil),
			outputWeights: mat.NewDense(dModel, hidden, randomArray(hidden*dModel, float64(hidden))),
			outputBias:    mat.NewDense(dModel, 1, nil),
		}

		gpt.blocks[i] = TransformerBlock{
			attn: attn,
			mlp:  mlp,
		}
	}
	return gpt
}

func NewAttention(dModel, nHeads int, lr float64) *Attention {
	if dModel%nHeads != 0 {
		panic("dModel must be divisible by nHeads")
	}
	dHead := dModel / nHeads
	attn := &Attention{
		H:            nHeads,
		dModel:       dModel,
		dHead:        dHead,
		learningRate: lr,
		Wquery:       make([]*mat.Dense, nHeads),
		Wkey:         make([]*mat.Dense, nHeads),
		Wvalue:       make([]*mat.Dense, nHeads),
		Q:            make([]*mat.Dense, nHeads),
		K:            make([]*mat.Dense, nHeads),
		V:            make([]*mat.Dense, nHeads),
		Scores:       make([]*mat.Dense, nHeads),
		A:            make([]*mat.Dense, nHeads),
		O:            make([]*mat.Dense, nHeads),
		maskCache:    make(map[int]*mat.Dense),
		parallel:     os.Getenv("HEAD_PAR") == "1",
	}
	for h := 0; h < nHeads; h++ {
		attn.Wquery[h] = mat.NewDense(dHead, dModel, randomArray(dHead*dModel, float64(dModel)))
		attn.Wkey[h] = mat.NewDense(dHead, dModel, randomArray(dHead*dModel, float64(dModel)))
		attn.Wvalue[h] = mat.NewDense(dHead, dModel, randomArray(dHead*dModel, float64(dModel)))
	}
	attn.Woutput = mat.NewDense(dModel, dModel, randomArray(dModel*dModel, float64(dModel)))
	return attn
}

// Block forward/backward with residuals.
func (b *TransformerBlock) Forward(X *mat.Dense) *mat.Dense {
	attnOut := b.attn.Forward(X)
	X = toDense(add(X, attnOut))
	mlpOut := b.mlp.Forward(X)
	X = toDense(add(X, mlpOut))
	return X
}

func (b *TransformerBlock) Backward(grad *mat.Dense) *mat.Dense {
	grad = expandGradToSeq(grad, b.mlp.lastInput)

	// Forward: X1 = X0  Attn(X0); Y = X1  MLP(X1)
	// Backward (top residual): dX1_total = grad (identity)  dMLP
	gradIntoX1FromMLP := b.mlp.Backward(grad)                // dL/dX1 via MLP
	gradIntoX1Total := toDense(add(grad, gradIntoX1FromMLP)) // + identity
	gradIntoX0FromAttn := b.attn.Backward(gradIntoX1Total)
	gradIntoX0Total := toDense(add(gradIntoX1Total, gradIntoX0FromAttn))
	return gradIntoX0Total

}

func (b *TransformerBlock) BackwardGradsOnly(grad *mat.Dense) (dX *mat.Dense,
	dWq, dWk, dWv []*mat.Dense, dWo *mat.Dense,
	dWhid, dWout *mat.Dense) {

	grad = expandGradToSeq(grad, b.mlp.lastInput)

	gradIntoX1FromMLP, dWhid, _, dWout, _ := b.mlp.BackwardGradsOnly(grad)
	gradIntoX1Total := toDense(add(grad, gradIntoX1FromMLP))

	dXattn, dWq, dWk, dWv, dWo := b.attn.BackwardGradsOnly(gradIntoX1Total)
	dX = toDense(add(gradIntoX1Total, dXattn))
	return
}

// ForwardLastWithKV: one-timestep forward using KV cache for attention.
// xLast: (dModel x 1). Returns yLast: (dModel x 1).
type BlockKV struct {
	attnKV AttnKV
}

func (b *TransformerBlock) ForwardLastWithKV(xLast *mat.Dense, kv *AttnKV) *mat.Dense {
	attnOut := b.attn.ForwardLastWithKV(xLast, kv) // (dModel x 1)
	x1 := toDense(add(xLast, attnOut))             // residual
	mlpOut := b.mlp.ForwardCol(x1)                 // (dModel x 1)
	y := toDense(add(x1, mlpOut))
	return y
}
