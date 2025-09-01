package main

import "gonum.org/v1/gonum/mat"

func (mlp *MLP) Forward(X *mat.Dense) *mat.Dense {
	mlp.lastInput = X
	hiddenLin := toDense(dot(mlp.hiddenWeights, X))      // (h x T)
	hiddenWithBias := addBias(hiddenLin, mlp.hiddenBias) // (h x T)
	mlp.hiddenOutputs = apply(sigmoid, hiddenWithBias).(*mat.Dense)
	finalLin := toDense(dot(mlp.outputWeights, mlp.hiddenOutputs)) // (d x T)
	finalWithBias := addBias(finalLin, mlp.outputBias)             // (d x T)
	mlp.finalOutputs = finalWithBias
	return mlp.finalOutputs
}

func (mlp *MLP) Backward(grad *mat.Dense) *mat.Dense {

	dX, dWhid, dbHidden, dWout, dbOut := mlp.BackwardGradsOnly(grad)
	lr := mlp.learningRate
	mlp.outputWeights = add(mlp.outputWeights, scale(-lr, dWout)).(*mat.Dense)
	mlp.outputBias = add(mlp.outputBias, scale(-lr, dbOut)).(*mat.Dense)
	mlp.hiddenWeights = add(mlp.hiddenWeights, scale(-lr, dWhid)).(*mat.Dense)
	mlp.hiddenBias = add(mlp.hiddenBias, scale(-lr, dbHidden)).(*mat.Dense)
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

	hiddenPre := toDense(dot(mlp.outputWeights.T(), grad))
	hiddenErrors := multiply(hiddenPre, sigmoidPrime(mlp.hiddenOutputs)).(*mat.Dense)

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
	h.Mul(mlp.hiddenWeights, xCol)      // (h x 1)
	hb := addBias(toDense(&h), mlp.hiddenBias)
	hs := apply(sigmoid, hb).(*mat.Dense)
	// out = W_out*hidden + b_out
	var o mat.Dense
	o.Mul(mlp.outputWeights, hs)
	ob := addBias(toDense(&o), mlp.outputBias)
	return ob
}
