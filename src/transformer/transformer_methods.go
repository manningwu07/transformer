package transformer

import (
	"math"
	"os"

	"github.com/manningwu07/GPT/optimizations"
	"github.com/manningwu07/GPT/params"
	"github.com/manningwu07/GPT/utils"
	"gonum.org/v1/gonum/mat"
)

type Transformer struct {
	Blocks []TransformerBlock
}

type TransformerBlock struct {
	Attn *Attention
	Mlp  *MLP
	Ln1  *optimizations.LayerNorm
	Ln2  *optimizations.LayerNorm
}

// Initalization

func CreateGPT(dModel, hidden, vocabSize int, AttnRate float64, MLPRate float64) Transformer {
	gpt := Transformer{
		Blocks: make([]TransformerBlock, params.Layers),
	}

	numHeads := utils.ChooseValidHeads(dModel, params.Config.NumHeads)

	for i := range params.Layers {
		attn := NewAttention(dModel, numHeads, AttnRate)

		mlp := &MLP{
			// MLP
			Inputs:        dModel,
			Hiddens:       hidden,
			Outputs:       dModel,
			LearningRate:  MLPRate,
			HiddenWeights: mat.NewDense(hidden, dModel, utils.RandomArray(dModel*hidden, float64(dModel))),
			HiddenBias:    mat.NewDense(hidden, 1, nil),
			OutputWeights: mat.NewDense(dModel, hidden, utils.RandomArray(hidden*dModel, float64(hidden))),
			OutputBias:    mat.NewDense(dModel, 1, nil),

			// Adam
			MHiddenW: mat.NewDense(hidden, dModel, nil),
			VHiddenW: mat.NewDense(hidden, dModel, nil),
			MHiddenB: mat.NewDense(hidden, 1, nil),
			VHiddenB: mat.NewDense(hidden, 1, nil),
			MOutputW: mat.NewDense(dModel, hidden, nil),
			VOutputW: mat.NewDense(dModel, hidden, nil),
			MOutputB: mat.NewDense(dModel, 1, nil),
			VOutputB: mat.NewDense(dModel, 1, nil),
		}

		gpt.Blocks[i] = TransformerBlock{
			Attn: attn,
			Mlp:  mlp,
			Ln1:  optimizations.NewLayerNorm(dModel, 1e-5, params.Config.NormLR),
			Ln2:  optimizations.NewLayerNorm(dModel, 1e-5, params.Config.NormLR),
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
		DModel:       dModel,
		DHead:        dHead,
		LearningRate: lr,
		Wquery:       make([]*mat.Dense, nHeads),
		Wkey:         make([]*mat.Dense, nHeads),
		Wvalue:       make([]*mat.Dense, nHeads),

		// Adam
		MWq: make([]*mat.Dense, nHeads),
		VWq: make([]*mat.Dense, nHeads),
		MWk: make([]*mat.Dense, nHeads),
		VWk: make([]*mat.Dense, nHeads),
		MWv: make([]*mat.Dense, nHeads),
		VWv: make([]*mat.Dense, nHeads),

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
		attn.Wquery[h] = mat.NewDense(dHead, dModel, utils.RandomArray(dHead*dModel, float64(dModel)))
		attn.Wkey[h] = mat.NewDense(dHead, dModel, utils.RandomArray(dHead*dModel, float64(dModel)))
		attn.Wvalue[h] = mat.NewDense(dHead, dModel, utils.RandomArray(dHead*dModel, float64(dModel)))

		attn.MWq[h] = mat.NewDense(dHead, dModel, nil)
		attn.VWq[h] = mat.NewDense(dHead, dModel, nil)
		attn.MWk[h] = mat.NewDense(dHead, dModel, nil)
		attn.VWk[h] = mat.NewDense(dHead, dModel, nil)
		attn.MWv[h] = mat.NewDense(dHead, dModel, nil)
		attn.VWv[h] = mat.NewDense(dHead, dModel, nil)
	}

	attn.Woutput = mat.NewDense(dModel, dModel, utils.RandomArray(dModel*dModel, float64(dModel)))
	attn.MWo = mat.NewDense(dModel, dModel, nil)
	attn.VWo = mat.NewDense(dModel, dModel, nil)
	return attn
}

// Block forward/backward with residuals.
func (b *TransformerBlock) Forward(X *mat.Dense) *mat.Dense {
	// Lazily init layer norms for test paths that construct Blocks manually.
	d, _ := X.Dims()
	b.EnsureNorms(d)

	x1 := b.Ln1.Forward(X)
	attnOut := b.Attn.Forward(x1)
	xRes := utils.ToDense(utils.Add(X, utils.Scale(1/math.Sqrt(2), attnOut)))
	x2 := b.Ln2.Forward(xRes)
	mlpOut := b.Mlp.Forward(x2)
	return utils.ToDense(utils.Add(xRes, utils.Scale(1/math.Sqrt(2), mlpOut)))
}

func (b *TransformerBlock) Backward(grad *mat.Dense) *mat.Dense {
	// Ensure norms exist (some tests bypass CreateGPT).
	if b.Mlp != nil {
		b.EnsureNorms(b.Mlp.Inputs)
	}

	// Y = xRes + c*MLP(x2); x2 = Ln2(xRes); xRes = X + c*Attn(Ln1(X))
	grad = utils.ExpandGradToSeq(grad, b.Mlp.lastInput) // dY shape -> (d x T)
	c := 1 / math.Sqrt(2)

	// MLP path: dL/d(MLP_out) = c * dL/dY
	dX2_fromMLP := b.Mlp.Backward(utils.ToDense(utils.Scale(c, grad))) // dL/dx2

	dXres_fromLn2 := b.Ln2.Backward(dX2_fromMLP)
	dXres_total := utils.ToDense(utils.Add(grad, dXres_fromLn2))
	dX1_fromAttn := b.Attn.Backward(utils.ToDense(utils.Scale(c, dXres_total))) // dL/d(Ln1(X))
	dX_fromLn1 := b.Ln1.Backward(dX1_fromAttn)

	return utils.ToDense(utils.Add(dXres_total, dX_fromLn1))

}

func (b *TransformerBlock) BackwardGradsOnly(grad *mat.Dense) (dX *mat.Dense,
	dWq, dWk, dWv []*mat.Dense, dWo *mat.Dense,
	dWhid, dWout *mat.Dense) {

	// Ensure norms for tests that build Blocks manually.
	if b.Mlp != nil {
		b.EnsureNorms(b.Mlp.Inputs)
	}
	grad = utils.ExpandGradToSeq(grad, b.Mlp.lastInput)
	c := 1 / math.Sqrt(2)

	// MLP path with residual scaling
	dX2_fromMLP, dWhid, _, dWout, _ := b.Mlp.BackwardGradsOnly(utils.ToDense(utils.Scale(c, grad)))
	dXres_fromLn2, _, _ := b.Ln2.BackwardGradsOnly(dX2_fromMLP)
	dXres_total := utils.ToDense(utils.Add(grad, dXres_fromLn2))
	dX1_fromAttn, dWq, dWk, dWv, dWo := b.Attn.BackwardGradsOnly(utils.ToDense(utils.Scale(c, dXres_total)))
	dX_fromLn1, _, _ := b.Ln1.BackwardGradsOnly(dX1_fromAttn)

	dX = utils.ToDense(utils.Add(dXres_total, dX_fromLn1))
	return dX, dWq, dWk, dWv, dWo, dWhid, dWout
}

func (b *TransformerBlock) ForwardLastWithKV(xLast *mat.Dense, kv *AttnKV) *mat.Dense {
	// Lazily init layer norms for test paths that construct Blocks manually.
	d, _ := xLast.Dims()
	b.EnsureNorms(d)

	c := 1 / math.Sqrt(2)
	n1 := b.Ln1.ForwardCol(xLast)
	attnOut := b.Attn.ForwardLastWithKV(n1, kv) // (dModel x 1)
	x1 := utils.ToDense(utils.Add(xLast, utils.Scale(c, attnOut)))
	n2 := b.Ln2.ForwardCol(x1)
	mlpOut := b.Mlp.ForwardCol(n2) // (dModel x 1)
	y := utils.ToDense(utils.Add(x1, utils.Scale(c, mlpOut)))
	return y
}

// ensureNorms lazily allocates LayerNorms if they are nil.
func (b *TransformerBlock) EnsureNorms(d int) {
	if b.Ln1 == nil {
		b.Ln1 = optimizations.NewLayerNorm(d, 1e-5, params.Config.NormLR)
	}
	if b.Ln2 == nil {
		b.Ln2 = optimizations.NewLayerNorm(d, 1e-5, params.Config.NormLR)
	}
}