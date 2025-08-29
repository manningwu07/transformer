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
    X      *mat.Dense    
    Q, K, V []*mat.Dense 
    Scores  []*mat.Dense
    A       []*mat.Dense
    O       []*mat.Dense
    O_cat    *mat.Dense   
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

func CreateGPT(input, hidden, output int, AttnRate float64, MLPRate float64) Transformer {
	gpt := Transformer{
		blocks: make([]TransformerBlock, layers),
	}

	for i := range layers {
		attn := NewAttention(input, 8, AttnRate)

		mlp := &MLP{
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

func NewAttention(dModel, nHeads int, lr float64) *Attention {
    if dModel % nHeads != 0 {
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
    }
    for h := 0; h < nHeads; h++ {
        attn.Wquery[h] = mat.NewDense(dHead, dModel, randomArray(dHead*dModel, float64(dModel)))
        attn.Wkey[h]   = mat.NewDense(dHead, dModel, randomArray(dHead*dModel, float64(dModel)))
        attn.Wvalue[h] = mat.NewDense(dHead, dModel, randomArray(dHead*dModel, float64(dModel)))
    }
    attn.Woutput = mat.NewDense(dModel, dModel, randomArray(dModel*dModel, float64(dModel)))
    return attn
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
    headsCat := mat.NewDense(attn.dModel, 1, nil)

    row := 0
    rescale := 1.0 / math.Sqrt(float64(attn.dHead))

    for h := 0; h < attn.H; h++ {
        Q := dot(attn.Wquery[h], X).(*mat.Dense) // (dHead x 1)
        K := dot(attn.Wkey[h], X).(*mat.Dense)
        V := dot(attn.Wvalue[h], X).(*mat.Dense)

        S := scale(rescale, dot(Q, K.T())).(*mat.Dense) // (dHead x dHead)
        A := RowSoftmax(S)
        O := dot(A, V).(*mat.Dense) // (dHead x 1)

        attn.Q[h], attn.K[h], attn.V[h] = Q, K, V
        attn.Scores[h], attn.A[h], attn.O[h] = S, A, O

        // concat into headsCat
        for i := 0; i < attn.dHead; i++ {
            headsCat.Set(row+i, 0, O.At(i, 0))
        }
        row += attn.dHead
    }
    attn.O_cat = headsCat
    Y := dot(attn.Woutput, headsCat).(*mat.Dense) // (dModel x 1)
    return Y
}

// Backward: computes grads and updates weights (SGD)
func (attn *Attention) Backward(dY *mat.Dense) *mat.Dense {
    dX, dWq, dWk, dWv, dWout := attn.BackwardGradsOnly(dY)

    // === Apply updates ===
    lr := attn.learningRate
    for h := 0; h < attn.H; h++ {
        attn.Wquery[h] = add(attn.Wquery[h], scale(-lr, dWq[h])).(*mat.Dense)
        attn.Wkey[h]   = add(attn.Wkey[h],   scale(-lr, dWk[h])).(*mat.Dense)
        attn.Wvalue[h] = add(attn.Wvalue[h], scale(-lr, dWv[h])).(*mat.Dense)
    }
    attn.Woutput = add(attn.Woutput, scale(-lr, dWout)).(*mat.Dense)

    return dX
}

// BackwardGradsOnly: computes grads but does NOT update weights
func (attn *Attention) BackwardGradsOnly(dY *mat.Dense) (
    dX *mat.Dense,
    dWq, dWk, dWv []*mat.Dense,
    dWout *mat.Dense,
) {
    dWq = make([]*mat.Dense, attn.H)
    dWk = make([]*mat.Dense, attn.H)
    dWv = make([]*mat.Dense, attn.H)

    // dY wrt Y = Wout * Ocat
    dWout = dot(dY, attn.O_cat.T()).(*mat.Dense)
    dOcat := dot(attn.Woutput.T(), dY).(*mat.Dense)

    dXtotal := mat.NewDense(attn.dModel, 1, nil)

    row := 0
    rescale := 1.0 / math.Sqrt(float64(attn.dHead))

    for h := 0; h < attn.H; h++ {
        // slice out this head’s portion of dOcat
        dO := mat.NewDense(attn.dHead, 1, nil)
        for i := 0; i < attn.dHead; i++ {
            dO.Set(i, 0, dOcat.At(row+i, 0))
        }
        row += attn.dHead

        // O = A V
        dA := dot(dO, attn.V[h].T()).(*mat.Dense)
        dV := dot(attn.A[h].T(), dO).(*mat.Dense)

        // softmax backward
        dS := softmaxBackward(dA, attn.A[h])

        // scores = Q K^T / sqrt(dHead)
        dQ := scale(rescale, dot(dS, attn.K[h])).(*mat.Dense)
        dK := scale(rescale, dot(dS.T(), attn.Q[h])).(*mat.Dense)

        // param grads
        dWq[h] = dot(dQ, attn.X.T()).(*mat.Dense)
        dWk[h] = dot(dK, attn.X.T()).(*mat.Dense)
        dWv[h] = dot(dV, attn.X.T()).(*mat.Dense)

        // input grads
        dXq := dot(attn.Wquery[h].T(), dQ).(*mat.Dense)
        dXk := dot(attn.Wkey[h].T(), dK).(*mat.Dense)
        dXv := dot(attn.Wvalue[h].T(), dV).(*mat.Dense)
        dXh := add(add(dXq, dXk), dXv).(*mat.Dense)
        dXtotal = add(dXtotal, dXh).(*mat.Dense)
    }

    return dXtotal, dWq, dWk, dWv, dWout
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
