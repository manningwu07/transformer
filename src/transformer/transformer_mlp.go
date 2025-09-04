package transformer

import (
	"github.com/manningwu07/GPT/optimizations"
	"github.com/manningwu07/GPT/params"
	"github.com/manningwu07/GPT/utils"
	"gonum.org/v1/gonum/mat"
)

type MLP struct {
	Inputs, Hiddens, Outputs  int
	HiddenWeights, HiddenBias *mat.Dense
	OutputWeights, OutputBias *mat.Dense
	LearningRate              float64

	// Adam
	T                  int
	MHiddenW, VHiddenW *mat.Dense
	MHiddenB, VHiddenB *mat.Dense
	MOutputW, VOutputW *mat.Dense
	MOutputB, VOutputB *mat.Dense

	// cache for backprop
	lastInput, hiddenPreAct, hiddenOutputs, finalOutputs *mat.Dense
}

func (mlp *MLP) Forward(X *mat.Dense) *mat.Dense {
	mlp.lastInput = X
	hiddenLin := utils.ToDense(utils.Dot(mlp.HiddenWeights, X)) // (h x T)
	hiddenWithBias := utils.AddBias(hiddenLin, mlp.HiddenBias)        // (h x T)
	mlp.hiddenPreAct = hiddenWithBias
	mlp.hiddenOutputs = utils.Apply(utils.GeluApply, hiddenWithBias).(*mat.Dense)
	finalLin := utils.ToDense(utils.Dot(mlp.OutputWeights, mlp.hiddenOutputs)) // (d x T)
	finalWithBias := utils.AddBias(finalLin, mlp.OutputBias)                         // (d x T)
	mlp.finalOutputs = finalWithBias
	return mlp.finalOutputs
}

func (mlp *MLP) Backward(grad *mat.Dense) *mat.Dense {

	dX, dWhid, dbHidden, dWout, dbOut := mlp.BackwardGradsOnly(grad)
	mlp.T++
	lr := mlp.LearningRate
	// Optional global clipping for this Module
	if params.Config.GradClip > 0 {
		s := utils.ClipGrads(params.Config.GradClip, dWout, dWhid, dbOut, dbHidden)
		if s < 1.0 && params.Config.Debug && mlp.T%params.Config.DebugEvery == 0 {
			utils.Debugf("MLP: clipped grads by %.4f at step %d", s, mlp.T)
		}
	}

	// AdamW: weight decay only on weights, not biases
	optimizations.AdamUpdateInPlace(mlp.OutputWeights, dWout, mlp.MOutputW, mlp.VOutputW,
		mlp.T, lr, params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps,
		params.Config.WeightDecay)
	optimizations.AdamUpdateInPlace(mlp.OutputBias, dbOut, mlp.MOutputB, mlp.VOutputB, mlp.T,
		lr, params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, 0.0)
	optimizations.AdamUpdateInPlace(mlp.HiddenWeights, dWhid, mlp.MHiddenW, mlp.VHiddenW,
		mlp.T, lr, params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps,
		params.Config.WeightDecay)
	optimizations.AdamUpdateInPlace(mlp.HiddenBias, dbHidden, mlp.MHiddenB, mlp.VHiddenB,
		mlp.T, lr, params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, 0.0)
	return dX
}

func (mlp *MLP) BackwardGradsOnly(grad *mat.Dense) (dX, dWhid, dbHidden, dWout, dbOut *mat.Dense) {

	grad = utils.ExpandGradToSeq(grad, mlp.lastInput)

	dWout = utils.Dot(grad, mlp.hiddenOutputs.T()).(*mat.Dense)
	// sum gradients over time for biases
	_, T := grad.Dims()
	dbOut = mat.NewDense(mlp.Outputs, 1, nil)
	for i := 0; i < mlp.Outputs; i++ {
		s := 0.0
		for t := 0; t < T; t++ {
			s += grad.At(i, t)
		}
		dbOut.Set(i, 0, s)
	}

	hiddenGradOut := utils.ToDense(utils.Dot(mlp.OutputWeights.T(), grad)) // dL/d(hidden_out)
	hiddenErrors := utils.Multiply(hiddenGradOut, utils.GeluPrime(mlp.hiddenPreAct)).(*mat.Dense)

	dWhid = utils.ToDense(utils.Dot(hiddenErrors, mlp.lastInput.T()))
	dbHidden = mat.NewDense(mlp.Hiddens, 1, nil)
	for i := 0; i < mlp.Hiddens; i++ {
		s := 0.0
		for t := 0; t < T; t++ {
			s += hiddenErrors.At(i, t)
		}
		dbHidden.Set(i, 0, s)
	}

	dX = utils.ToDense(utils.Dot(mlp.HiddenWeights.T(), hiddenErrors))
	return dX, dWhid, dbHidden, dWout, dbOut
}

// ForwardCol: one column only, returns (dModel x 1)
func (mlp *MLP) ForwardCol(xCol *mat.Dense) *mat.Dense {
	// hidden = sigMoid(W_hid*x + b_hid)
	var h mat.Dense
	h.Mul(mlp.HiddenWeights, xCol) // (h x 1)
	hb := utils.AddBias(utils.ToDense(&h), mlp.HiddenBias)
	hs := utils.Apply(utils.GeluApply, hb).(*mat.Dense)
	// out = W_out*hidden + b_out
	var o mat.Dense
	o.Mul(mlp.OutputWeights, hs)
	ob := utils.AddBias(utils.ToDense(&o), mlp.OutputBias)
	return ob
}
