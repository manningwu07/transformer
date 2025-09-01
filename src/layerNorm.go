
package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

type LayerNorm struct {
	d            int
	eps          float64
	gamma        *mat.Dense // (d x 1)
	beta         *mat.Dense // (d x 1)
	learningRate float64

	// cache
	lastInput *mat.Dense // (d x T)
	xhat      *mat.Dense // (d x T)
	invStd    []float64  // per column

	// Adam state
	t           int
	mGamma, vGamma *mat.Dense
	mBeta, vBeta   *mat.Dense
}

func NewLayerNorm(d int, eps float64, lr float64) *LayerNorm {
	g := onesLike(mat.NewDense(d, 1, nil))
	b := mat.NewDense(d, 1, nil)
	return &LayerNorm{
		d:            d,
		eps:          eps,
		gamma:        g,
		beta:         b,
		learningRate: lr,
		mGamma:       mat.NewDense(d, 1, nil),
		vGamma:       mat.NewDense(d, 1, nil),
		mBeta:        mat.NewDense(d, 1, nil),
		vBeta:        mat.NewDense(d, 1, nil),
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
		istd := 1.0 / math.Sqrt(v+ln.eps)
		inv[t] = istd
		// normalize and affine
		for i := 0; i < d; i++ {
			n := (X.At(i, t) - mu) * istd
			xhat.Set(i, t, n)
			val := ln.gamma.At(i, 0)*n + ln.beta.At(i, 0)
			out.Set(i, t, val)
		}
	}
	ln.lastInput = X
	ln.xhat = xhat
	ln.invStd = inv
	return out
}

// Backward applies Adam updates to gamma/beta and returns dX.
func (ln *LayerNorm) Backward(dY *mat.Dense) *mat.Dense {
	dX, dGamma, dBeta := ln.BackwardGradsOnly(dY)
	ln.t++
	adamUpdateInPlace(
		ln.gamma, dGamma, ln.mGamma, ln.vGamma, ln.t, ln.learningRate,
		config.AdamBeta1, config.AdamBeta2, config.AdamEps,
	)
	adamUpdateInPlace(
		ln.beta, dBeta, ln.mBeta, ln.vBeta, ln.t, ln.learningRate,
		config.AdamBeta1, config.AdamBeta2, config.AdamEps,
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
			sumDG += dY.At(i, t) * ln.xhat.At(i, t)
			sumDB += dY.At(i, t)
		}
		dGamma.Set(i, 0, sumDG)
		dBeta.Set(i, 0, sumDB)
	}

	// dX (per column)
	dX = mat.NewDense(d, T, nil)
	for t := 0; t < T; t++ {
		istd := ln.invStd[t]
		// precompute sums
		sum1 := 0.0
		sum2 := 0.0
		for i := 0; i < d; i++ {
			gy := dY.At(i, t) * ln.gamma.At(i, 0)
			sum1 += gy
			sum2 += gy * ln.xhat.At(i, t)
		}
		for i := 0; i < d; i++ {
			gy := dY.At(i, t) * ln.gamma.At(i, 0)
			dxi := (float64(d)*gy - sum1 - ln.xhat.At(i, t)*sum2) * (istd / float64(d))
			dX.Set(i, t, dxi)
		}
	}
	return dX, dGamma, dBeta
}

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
	istd := 1.0 / math.Sqrt(v+ln.eps)
	out := mat.NewDense(d, 1, nil)
	for i := 0; i < d; i++ {
		n := (x.At(i, 0) - mu) * istd
		out.Set(i, 0, ln.gamma.At(i, 0)*n+ln.beta.At(i, 0))
	}
	return out
}