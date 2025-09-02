package main

import (
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
)

type Transformer struct {
	blocks []TransformerBlock
}

type TransformerBlock struct {
	attn *Attention
	mlp  *MLP
	ln1  *LayerNorm
	ln2  *LayerNorm
}

// Initalization

func CreateGPT(dModel, hidden, vocabSize int, AttnRate float64, MLPRate float64) Transformer {
	gpt := Transformer{
		blocks: make([]TransformerBlock, layers),
	}

	numHeads := chooseValidHeads(dModel, config.NumHeads)

	for i := range layers {
		attn := NewAttention(dModel, numHeads, AttnRate)

		mlp := &MLP{
			// MLP
			inputs:        dModel,
			hiddens:       hidden,
			outputs:       dModel,
			learningRate:  MLPRate,
			hiddenWeights: mat.NewDense(hidden, dModel, randomArray(dModel*hidden, float64(dModel))),
			hiddenBias:    mat.NewDense(hidden, 1, nil),
			outputWeights: mat.NewDense(dModel, hidden, randomArray(hidden*dModel, float64(hidden))),
			outputBias:    mat.NewDense(dModel, 1, nil),

			// Adam
			mHiddenW: mat.NewDense(hidden, dModel, nil),
			vHiddenW: mat.NewDense(hidden, dModel, nil),
			mHiddenB: mat.NewDense(hidden, 1, nil),
			vHiddenB: mat.NewDense(hidden, 1, nil),
			mOutputW: mat.NewDense(dModel, hidden, nil),
			vOutputW: mat.NewDense(dModel, hidden, nil),
			mOutputB: mat.NewDense(dModel, 1, nil),
			vOutputB: mat.NewDense(dModel, 1, nil),
		}

		gpt.blocks[i] = TransformerBlock{
			attn: attn,
			mlp:  mlp,
			ln1:  NewLayerNorm(dModel, 1e-5, config.NormLR),
			ln2:  NewLayerNorm(dModel, 1e-5, config.NormLR),
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

		// Adam
		mWq: make([]*mat.Dense, nHeads),
		vWq: make([]*mat.Dense, nHeads),
		mWk: make([]*mat.Dense, nHeads),
		vWk: make([]*mat.Dense, nHeads),
		mWv: make([]*mat.Dense, nHeads),
		vWv: make([]*mat.Dense, nHeads),

		// cache
		Q:         make([]*mat.Dense, nHeads),
		K:         make([]*mat.Dense, nHeads),
		V:         make([]*mat.Dense, nHeads),
		Scores:    make([]*mat.Dense, nHeads),
		A:         make([]*mat.Dense, nHeads),
		O:         make([]*mat.Dense, nHeads),
		maskCache: make(map[int]*mat.Dense),
		parallel:  os.Getenv("HEAD_PAR") == "1",
	}
	for h := 0; h < nHeads; h++ {
		attn.Wquery[h] = mat.NewDense(dHead, dModel, randomArray(dHead*dModel, float64(dModel)))
		attn.Wkey[h] = mat.NewDense(dHead, dModel, randomArray(dHead*dModel, float64(dModel)))
		attn.Wvalue[h] = mat.NewDense(dHead, dModel, randomArray(dHead*dModel, float64(dModel)))

		attn.mWq[h] = mat.NewDense(dHead, dModel, nil)
		attn.vWq[h] = mat.NewDense(dHead, dModel, nil)
		attn.mWk[h] = mat.NewDense(dHead, dModel, nil)
		attn.vWk[h] = mat.NewDense(dHead, dModel, nil)
		attn.mWv[h] = mat.NewDense(dHead, dModel, nil)
		attn.vWv[h] = mat.NewDense(dHead, dModel, nil)
	}

	attn.Woutput = mat.NewDense(dModel, dModel, randomArray(dModel*dModel, float64(dModel)))
	attn.mWo = mat.NewDense(dModel, dModel, nil)
	attn.vWo = mat.NewDense(dModel, dModel, nil)
	return attn
}

// Block forward/backward with residuals.
func (b *TransformerBlock) Forward(X *mat.Dense) *mat.Dense {
	// Lazily init layer norms for test paths that construct blocks manually.
	d, _ := X.Dims()
	b.ensureNorms(d)

	x1 := b.ln1.Forward(X)
	attnOut := b.attn.Forward(x1)
	xRes := toDense(add(X, scale(1/math.Sqrt(2), attnOut)))
	x2 := b.ln2.Forward(xRes)
	mlpOut := b.mlp.Forward(x2)
	return toDense(add(xRes, scale(1/math.Sqrt(2), mlpOut)))
}

func (b *TransformerBlock) Backward(grad *mat.Dense) *mat.Dense {
	// Ensure norms exist (some tests bypass CreateGPT).
	if b.mlp != nil {
		b.ensureNorms(b.mlp.inputs)
	}

	// Y = xRes + c*MLP(x2); x2 = LN2(xRes); xRes = X + c*Attn(LN1(X))
	grad = expandGradToSeq(grad, b.mlp.lastInput) // dY shape -> (d x T)
	c := 1 / math.Sqrt(2)

	// MLP path: dL/d(MLP_out) = c * dL/dY
	dX2_fromMLP := b.mlp.Backward(toDense(scale(c, grad))) // dL/dx2

	dXres_fromLN2 := b.ln2.Backward(dX2_fromMLP)
	dXres_total := toDense(add(grad, dXres_fromLN2))
	dX1_fromAttn := b.attn.Backward(toDense(scale(c, dXres_total))) // dL/d(LN1(X))
	dX_fromLN1 := b.ln1.Backward(dX1_fromAttn)

	return toDense(add(dXres_total, dX_fromLN1))

}

func (b *TransformerBlock) BackwardGradsOnly(grad *mat.Dense) (dX *mat.Dense,
	dWq, dWk, dWv []*mat.Dense, dWo *mat.Dense,
	dWhid, dWout *mat.Dense) {

	// Ensure norms for tests that build blocks manually.
	if b.mlp != nil {
		b.ensureNorms(b.mlp.inputs)
	}
	grad = expandGradToSeq(grad, b.mlp.lastInput)
	c := 1 / math.Sqrt(2)

	// MLP path with residual scaling
	dX2_fromMLP, dWhid, _, dWout, _ := b.mlp.BackwardGradsOnly(toDense(scale(c, grad)))
	dXres_fromLN2, _, _ := b.ln2.BackwardGradsOnly(dX2_fromMLP)
	dXres_total := toDense(add(grad, dXres_fromLN2))
	dX1_fromAttn, dWq, dWk, dWv, dWo := b.attn.BackwardGradsOnly(toDense(scale(c, dXres_total)))
	dX_fromLN1, _, _ := b.ln1.BackwardGradsOnly(dX1_fromAttn)

	dX = toDense(add(dXres_total, dX_fromLN1))
	return dX, dWq, dWk, dWv, dWo, dWhid, dWout
}

func (b *TransformerBlock) ForwardLastWithKV(xLast *mat.Dense, kv *AttnKV) *mat.Dense {
	// Lazily init layer norms for test paths that construct blocks manually.
	d, _ := xLast.Dims()
	b.ensureNorms(d)

	c := 1 / math.Sqrt(2)
	n1 := b.ln1.ForwardCol(xLast)
	attnOut := b.attn.ForwardLastWithKV(n1, kv) // (dModel x 1)
	x1 := toDense(add(xLast, scale(c, attnOut)))
	n2 := b.ln2.ForwardCol(x1)
	mlpOut := b.mlp.ForwardCol(n2) // (dModel x 1)
	y := toDense(add(x1, scale(c, mlpOut)))
	return y
}
