package main

import (
	// "encoding/csv"
	// "fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	// "math/rand"
	// "os"
	// "strconv"
	// "time"
)

type Transformer struct {
	blocks []TransformerBlock
}

type TransformerBlock struct {
	attn Attention
	mlp  MLP
}

type Attention struct {
	Wquery       *mat.Dense
	Wkey         *mat.Dense
	Wvalue       *mat.Dense
	Woutput      *mat.Dense
	learningRate float64

	// cache for backprop
	X       *mat.Dense
	Q, K, V *mat.Dense
	scores  *mat.Dense
	A       *mat.Dense
	O       *mat.Dense
}

type MLP struct {
	inputs, hiddens, outputs  int
	hiddenWeights, hiddenBias *mat.Dense
	outputWeights, outputBias *mat.Dense
	learningRate              float64

	// cache for backprop
	lastInput, hiddenOutputs, finalOutputs *mat.Dense
}

func CreateGPT(input, hidden, output int, AttnRate float64, MLPRate float64) Transformer {
	gpt := Transformer{
		blocks: make([]TransformerBlock, layers),
	}

	for i := range layers {
		attn := Attention{
			Wquery:       mat.NewDense(hidden, input, randomArray(input*hidden, float64(input))),
			Wkey:         mat.NewDense(hidden, input, randomArray(input*hidden, float64(input))),
			Wvalue:       mat.NewDense(hidden, input, randomArray(input*hidden, float64(input))),
			Woutput:      mat.NewDense(output, hidden, randomArray(hidden*output, float64(hidden))),
			learningRate: AttnRate,
		}

		mlp := MLP{
			inputs:        input,
			hiddens:       hidden,
			outputs:       output,
			learningRate:  MLPRate,
			hiddenWeights: mat.NewDense(hidden, input, randomArray(input*hidden, float64(input))),
			hiddenBias:    mat.NewDense(hidden, 1, nil),
			outputWeights: mat.NewDense(output, hidden, randomArray(hidden*output, float64(hidden))),
			outputBias:    mat.NewDense(output, 1, nil),
		}

		gpt.blocks[i] = TransformerBlock{
			attn: attn,
			mlp:  mlp,
		}
	}
	return gpt
}

// Block forward/backward with residuals.
func (b *TransformerBlock) Forward(X *mat.Dense) *mat.Dense {
	attnOut := b.attn.Forward(X)
	X = add(X, attnOut).(*mat.Dense)
	mlpOut := b.mlp.Forward(X)
	X = add(X, mlpOut).(*mat.Dense)
	return X
}

func (b *TransformerBlock) Backward(grad *mat.Dense) *mat.Dense {
    // Set this up concurrently
	grad = b.mlp.Backward(grad)
	grad = b.attn.Backward(grad)
	return grad
}
// Attention forward/backward.
func (attn *Attention) Forward(X *mat.Dense) *mat.Dense {
	attn.X = X
	attn.Q = dot(attn.Wquery, X).(*mat.Dense) // [d x 1]
	attn.K = dot(attn.Wkey, X).(*mat.Dense)   // [d x 1]
	attn.V = dot(attn.Wvalue, X).(*mat.Dense) // [d x 1]

	dk, _ := attn.K.Dims() // key dimension = rows
	attn.scores = scale(1.0/math.Sqrt(float64(dk)),
		dot(attn.Q, attn.K.T()),
	).(*mat.Dense) // [d x d]

	attn.A = RowSoftmax(attn.scores)    // row-wise
	attn.O = dot(attn.A, attn.V).(*mat.Dense) // [d x 1]
	Y := dot(attn.Woutput, attn.O).(*mat.Dense)
	return Y
}

func (attn *Attention) Backward(dY *mat.Dense) *mat.Dense {
	// Y = Wout * O
	dWout := dot(dY, attn.O.T()).(*mat.Dense)
	dO := dot(attn.Woutput.T(), dY).(*mat.Dense)

	// O = A * V
	dA := dot(dO, attn.V.T()).(*mat.Dense)
	dV := dot(attn.A.T(), dO).(*mat.Dense)

	// softmax backward (row-wise)
	dS := softmaxBackward(dA, attn.A)

	// scores = (Q K^T)/sqrt(dk)
	dk, _ := attn.K.Dims()
	scaleFactor := 1.0 / math.Sqrt(float64(dk))
	dQ := dot(dS, attn.K).(*mat.Dense)
	dQ = scale(scaleFactor, dQ).(*mat.Dense)
	dK := dot(dS.T(), attn.Q).(*mat.Dense)
	dK = scale(scaleFactor, dK).(*mat.Dense)

	// linear projections
	dWq := dot(dQ, attn.X.T()).(*mat.Dense)
	dWk := dot(dK, attn.X.T()).(*mat.Dense)
	dWv := dot(dV, attn.X.T()).(*mat.Dense)

	dXq := dot(attn.Wquery.T(), dQ).(*mat.Dense)
	dXk := dot(attn.Wkey.T(), dK).(*mat.Dense)
	dXv := dot(attn.Wvalue.T(), dV).(*mat.Dense)
	dX := add(add(dXq, dXk), dXv).(*mat.Dense)

	// SGD update
	lr := attn.learningRate
	attn.Woutput = add(attn.Woutput, scale(-lr, dWout)).(*mat.Dense)
	attn.Wquery = add(attn.Wquery, scale(-lr, dWq)).(*mat.Dense)
	attn.Wkey = add(attn.Wkey, scale(-lr, dWk)).(*mat.Dense)
	attn.Wvalue = add(attn.Wvalue, scale(-lr, dWv)).(*mat.Dense)

	return dX
}

// BackwardGradsOnly: same as Backward but returns parameter grads without updating.
func (attn *Attention) BackwardGradsOnly(dY *mat.Dense) (
	dX, dWq, dWk, dWv, dWout *mat.Dense,
) {
	// Y = Wout * O
	dWout = dot(dY, attn.O.T()).(*mat.Dense)
	dO := dot(attn.Woutput.T(), dY).(*mat.Dense)

	// O = A * V
	dA := dot(dO, attn.V.T()).(*mat.Dense)
	dV := dot(attn.A.T(), dO).(*mat.Dense)

	// softmax backward
	dS := softmaxBackward(dA, attn.A)

	dk, _ := attn.K.Dims()
	scaleFactor := 1.0 / math.Sqrt(float64(dk))
	dQ := scale(scaleFactor, dot(dS, attn.K)).(*mat.Dense)
	dK := scale(scaleFactor, dot(dS.T(), attn.Q)).(*mat.Dense)

	dWq = dot(dQ, attn.X.T()).(*mat.Dense)
	dWk = dot(dK, attn.X.T()).(*mat.Dense)
	dWv = dot(dV, attn.X.T()).(*mat.Dense)

	dXq := dot(attn.Wquery.T(), dQ).(*mat.Dense)
	dXk := dot(attn.Wkey.T(), dK).(*mat.Dense)
	dXv := dot(attn.Wvalue.T(), dV).(*mat.Dense)
	dX = add(add(dXq, dXk), dXv).(*mat.Dense)
	return
}

// MLP functions

func (mlp *MLP) Forward(X *mat.Dense) *mat.Dense {
	mlp.lastInput = X
	hiddenInputs := add(dot(mlp.hiddenWeights, X), mlp.hiddenBias)
	mlp.hiddenOutputs = apply(sigmoid, hiddenInputs).(*mat.Dense)
	finalInputs := add(dot(mlp.outputWeights, mlp.hiddenOutputs), mlp.outputBias)
	mlp.finalOutputs = finalInputs.(*mat.Dense) // logits
	return mlp.finalOutputs
}

func (mlp *MLP) Backward(grad *mat.Dense) *mat.Dense {
	dWout := dot(grad, mlp.hiddenOutputs.T()).(*mat.Dense)
	dbOut := grad

	hiddenPre := dot(mlp.outputWeights.T(), grad).(*mat.Dense)
	hiddenErrors := multiply(hiddenPre, sigmoidPrime(mlp.hiddenOutputs)).(*mat.Dense)

	dWhid := dot(hiddenErrors, mlp.lastInput.T()).(*mat.Dense)
	dbHidden := hiddenErrors

	lr := mlp.learningRate
	mlp.outputWeights = add(mlp.outputWeights, scale(-lr, dWout)).(*mat.Dense)
	mlp.outputBias = add(mlp.outputBias, scale(-lr, dbOut)).(*mat.Dense)
	mlp.hiddenWeights = add(mlp.hiddenWeights, scale(-lr, dWhid)).(*mat.Dense)
	mlp.hiddenBias = add(mlp.hiddenBias, scale(-lr, dbHidden)).(*mat.Dense)

	dX := dot(mlp.hiddenWeights.T(), hiddenErrors).(*mat.Dense)
	return dX
}
