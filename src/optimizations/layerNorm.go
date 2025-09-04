package optimizations

import (
	"math"

	"github.com/manningwu07/GPT/params"
	"github.com/manningwu07/GPT/utils"
	"gonum.org/v1/gonum/mat"
)

type LayerNorm struct {
	D            int
	Eps          float64
	Gamma        *mat.Dense // (d x 1)
	Beta         *mat.Dense // (d x 1)
	LearningRate float64

	// cache
	LastInput *mat.Dense // (d x T)
	Xhat      *mat.Dense // (d x T)
	InvStd    []float64  // per column

	// Adam state
	T           int
	MGamma, VGamma *mat.Dense
	MBeta, VBeta   *mat.Dense
}

func NewLayerNorm(d int, eps float64, lr float64) *LayerNorm {
	g := utils.OnesLike(mat.NewDense(d, 1, nil))
	b := mat.NewDense(d, 1, nil)
	return &LayerNorm{
		D:            d,
		Eps:          eps,
		Gamma:        g,
		Beta:         b,
		LearningRate: lr,
		MGamma:       mat.NewDense(d, 1, nil),
		VGamma:       mat.NewDense(d, 1, nil),
		MBeta:        mat.NewDense(d, 1, nil),
		VBeta:        mat.NewDense(d, 1, nil),
	}
}

func (ln *LayerNorm) Forward(X *mat.Dense) *mat.Dense {
	d, T := X.Dims()
	out := mat.NewDense(d, T, nil)
	xhat := mat.NewDense(d, T, nil)
	inv := make([]float64, T)
	for t := 0; t < T; t++ {
		// mean over rows
		mu := 0.0
		for i := 0; i < d; i++ {
			mu += X.At(i, t)
		}
		mu /= float64(d)
		// variance
		var v float64
		for i := 0; i < d; i++ {
			diff := X.At(i, t) - mu
			v += diff * diff
		}
		v /= float64(d)
		istd := 1.0 / math.Sqrt(v+ln.Eps)
		inv[t] = istd
		// normalize and affine
		for i := 0; i < d; i++ {
			n := (X.At(i, t) - mu) * istd
			xhat.Set(i, t, n)
			val := ln.Gamma.At(i, 0)*n + ln.Beta.At(i, 0)
			out.Set(i, t, val)
		}
	}
	ln.LastInput = X
	ln.Xhat = xhat
	ln.InvStd = inv
	return out
}

// Backward applies Adam updates to gamma/beta and returns dX.
func (ln *LayerNorm) Backward(dY *mat.Dense) *mat.Dense {
	dX, dGamma, dBeta := ln.BackwardGradsOnly(dY)
	ln.T++
	AdamUpdateInPlace(
		ln.Gamma, dGamma, ln.MGamma, ln.VGamma, ln.T, ln.LearningRate,
        params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, 0.0,
	)
	AdamUpdateInPlace(
		ln.Beta, dBeta, ln.MBeta, ln.VBeta, ln.T, ln.LearningRate,
        params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, 0.0,
	)
	return dX
}


func (ln *LayerNorm) BackwardGradsOnly(dY *mat.Dense) (dX, dGamma, dBeta *mat.Dense) {
	d, T := dY.Dims()
	// grads for gamma/beta
	dGamma = mat.NewDense(d, 1, nil)
	dBeta = mat.NewDense(d, 1, nil)
	for i := 0; i < d; i++ {
		sumDG := 0.0
		sumDB := 0.0
		for t := 0; t < T; t++ {
			sumDG += dY.At(i, t) * ln.Xhat.At(i, t)
			sumDB += dY.At(i, t)
		}
		dGamma.Set(i, 0, sumDG)
		dBeta.Set(i, 0, sumDB)
	}

	// dX (per column)
	dX = mat.NewDense(d, T, nil)
	for t := 0; t < T; t++ {
		istd := ln.InvStd[t]
		// precompute sums
		sum1 := 0.0
		sum2 := 0.0
		for i := 0; i < d; i++ {
			gy := dY.At(i, t) * ln.Gamma.At(i, 0)
			sum1 += gy
			sum2 += gy * ln.Xhat.At(i, t)
		}
		for i := 0; i < d; i++ {
			gy := dY.At(i, t) * ln.Gamma.At(i, 0)
			dxi := (float64(d)*gy - sum1 - ln.Xhat.At(i, t)*sum2) * (istd / float64(d))
			dX.Set(i, t, dxi)
		}
	}
	return dX, dGamma, dBeta
}


// -----Helpers-----

// ForwardCol for inference (d x 1)
func (ln *LayerNorm) ForwardCol(x *mat.Dense) *mat.Dense {
	d, c := x.Dims()
	if c != 1 {
		panic("LayerNorm.ForwardCol expects (d x 1)")
	}
	mu := 0.0
	for i := 0; i < d; i++ { mu += x.At(i, 0) }
	mu /= float64(d)
	var v float64
	for i := 0; i < d; i++ {
		diff := x.At(i, 0) - mu
		v += diff * diff
	}
	v /= float64(d)
	istd := 1.0 / math.Sqrt(v+ln.Eps)
	out := mat.NewDense(d, 1, nil)
	for i := 0; i < d; i++ {
		n := (x.At(i, 0) - mu) * istd
		out.Set(i, 0, ln.Gamma.At(i, 0)*n+ln.Beta.At(i, 0))
	}
	return out
}