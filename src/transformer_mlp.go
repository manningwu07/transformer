package main

import "gonum.org/v1/gonum/mat"

type MLP struct {
	inputs, hiddens, outputs  int
	hiddenWeights, hiddenBias *mat.Dense
	outputWeights, outputBias *mat.Dense
	learningRate              float64

	// Adam
	t                  int
	mHiddenW, vHiddenW *mat.Dense
	mHiddenB, vHiddenB *mat.Dense
	mOutputW, vOutputW *mat.Dense
	mOutputB, vOutputB *mat.Dense

	// cache for backprop
	lastInput, hiddenPreAct, hiddenOutputs, finalOutputs *mat.Dense
}

func (mlp *MLP) Forward(X *mat.Dense) *mat.Dense {
	mlp.lastInput = X
	hiddenLin := toDense(dot(mlp.hiddenWeights, X))      // (h x T)
	hiddenWithBias := addBias(hiddenLin, mlp.hiddenBias) // (h x T)
	mlp.hiddenPreAct = hiddenWithBias
	mlp.hiddenOutputs = apply(geluApply, hiddenWithBias).(*mat.Dense)
	finalLin := toDense(dot(mlp.outputWeights, mlp.hiddenOutputs)) // (d x T)
	finalWithBias := addBias(finalLin, mlp.outputBias)             // (d x T)
	mlp.finalOutputs = finalWithBias
	return mlp.finalOutputs
}

func (mlp *MLP) Backward(grad *mat.Dense) *mat.Dense {

	dX, dWhid, dbHidden, dWout, dbOut := mlp.BackwardGradsOnly(grad)
	mlp.t++
	lr := mlp.learningRate
    // Optional global clipping for this module
    if config.GradClip > 0 {
        s := clipGrads(config.GradClip, dWout, dWhid, dbOut, dbHidden)
        if s < 1.0 && config.Debug && mlp.t%config.DebugEvery == 0 {
            debugf("MLP: clipped grads by %.4f at step %d", s, mlp.t)
        }
    }

    // AdamW: weight decay only on weights, not biases
    adamUpdateInPlace(mlp.outputWeights, dWout, mlp.mOutputW, mlp.vOutputW,
        mlp.t, lr, config.AdamBeta1, config.AdamBeta2, config.AdamEps,
        config.WeightDecay)
    adamUpdateInPlace(mlp.outputBias, dbOut, mlp.mOutputB, mlp.vOutputB, mlp.t,
        lr, config.AdamBeta1, config.AdamBeta2, config.AdamEps, 0.0)
    adamUpdateInPlace(mlp.hiddenWeights, dWhid, mlp.mHiddenW, mlp.vHiddenW,
        mlp.t, lr, config.AdamBeta1, config.AdamBeta2, config.AdamEps,
        config.WeightDecay)
    adamUpdateInPlace(mlp.hiddenBias, dbHidden, mlp.mHiddenB, mlp.vHiddenB,
        mlp.t, lr, config.AdamBeta1, config.AdamBeta2, config.AdamEps, 0.0)
	return dX
}

func (mlp *MLP) BackwardGradsOnly(grad *mat.Dense) (dX, dWhid, dbHidden, dWout, dbOut *mat.Dense) {

	grad = expandGradToSeq(grad, mlp.lastInput)

	dWout = dot(grad, mlp.hiddenOutputs.T()).(*mat.Dense)
	// sum gradients over time for biases
	_, T := grad.Dims()
	dbOut = mat.NewDense(mlp.outputs, 1, nil)
	for i := 0; i < mlp.outputs; i++ {
		s := 0.0
		for t := 0; t < T; t++ {
			s += grad.At(i, t)
		}
		dbOut.Set(i, 0, s)
	}

	hiddenGradOut := toDense(dot(mlp.outputWeights.T(), grad)) // dL/d(hidden_out)
	hiddenErrors := multiply(hiddenGradOut, geluPrime(mlp.hiddenPreAct)).(*mat.Dense)

	dWhid = toDense(dot(hiddenErrors, mlp.lastInput.T()))
	dbHidden = mat.NewDense(mlp.hiddens, 1, nil)
	for i := 0; i < mlp.hiddens; i++ {
		s := 0.0
		for t := 0; t < T; t++ {
			s += hiddenErrors.At(i, t)
		}
		dbHidden.Set(i, 0, s)
	}

	dX = toDense(dot(mlp.hiddenWeights.T(), hiddenErrors))
	return dX, dWhid, dbHidden, dWout, dbOut
}

// ForwardCol: one column only, returns (dModel x 1)
func (mlp *MLP) ForwardCol(xCol *mat.Dense) *mat.Dense {
	// hidden = sigmoid(W_hid*x + b_hid)
	var h mat.Dense
	h.Mul(mlp.hiddenWeights, xCol) // (h x 1)
	hb := addBias(toDense(&h), mlp.hiddenBias)
	hs := apply(geluApply, hb).(*mat.Dense)
	// out = W_out*hidden + b_out
	var o mat.Dense
	o.Mul(mlp.outputWeights, hs)
	ob := addBias(toDense(&o), mlp.outputBias)
	return ob
}
