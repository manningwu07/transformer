package main

import (
	"math"
	"sync"

	"gonum.org/v1/gonum/mat"
)

type Attention struct {
	H            int
	dModel       int
	dHead        int
	Wquery       []*mat.Dense
	Wkey         []*mat.Dense
	Wvalue       []*mat.Dense
	Woutput      *mat.Dense
	learningRate float64

	// Adam
	t        int
	mWq, vWq []*mat.Dense
	mWk, vWk []*mat.Dense
	mWv, vWv []*mat.Dense
	mWo, vWo *mat.Dense

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

// Attention forward/backward.
func (attn *Attention) Forward(X *mat.Dense) *mat.Dense {
	attn.X = X
	_, T := X.Dims() // T = number of columns (sequence length)
	headsCat := mat.NewDense(attn.dModel, T, nil)


 	rescale := 1.0 / math.Sqrt(float64(attn.dHead))
	// cache mask by T
	mask, ok := attn.maskCache[T]
	if !ok {
		mask = causalMask(T)
		attn.maskCache[T] = mask
	}

	// prepare per-head scratch resized once per T
	if attn.lastT != T {
		for h := 0; h < attn.H; h++ {
			attn.Q[h] = mat.NewDense(attn.dHead, T, nil)
			attn.K[h] = mat.NewDense(attn.dHead, T, nil)
			attn.V[h] = mat.NewDense(attn.dHead, T, nil)
			attn.Scores[h] = mat.NewDense(T, T, nil)
			attn.O[h] = mat.NewDense(attn.dHead, T, nil)
			attn.A[h] = nil // will be set fresh (RowSoftmaxMasked allocates)
		}
		attn.lastT = T
	}

	work := func(h int) {
		// Q,K,V
		attn.Q[h].Mul(attn.Wquery[h], X)
		attn.K[h].Mul(attn.Wkey[h], X)
		attn.V[h].Mul(attn.Wvalue[h], X)
		// S = (Q^T K)/sqrt
		attn.Scores[h].Mul(attn.Q[h].T(), attn.K[h])
		attn.Scores[h].Scale(rescale, attn.Scores[h])
		// A
		attn.A[h] = RowSoftmaxMasked(attn.Scores[h], mask)
		// O = V * A^T
		attn.O[h].Mul(attn.V[h], attn.A[h].T())
		// concat into headsCat rows
		base := h * attn.dHead
		dst := headsCat.Slice(base, base+attn.dHead, 0, T).(*mat.Dense)
		dst.Copy(attn.O[h])
	}
	if attn.parallel && attn.H > 1 {
		var wg sync.WaitGroup
		wg.Add(attn.H)
		for h := 0; h < attn.H; h++ {
			hh := h
			go func() { defer wg.Done(); work(hh) }()
		}
		wg.Wait()
	} else {
		for h := 0; h < attn.H; h++ { work(h) }
	}
	attn.O_cat = headsCat
	Y := toDense(dot(attn.Woutput, headsCat)) // (dModel x 1)
	return Y
}

// Backward: computes grads and updates weights (SGD)
func (attn *Attention) Backward(dY *mat.Dense) *mat.Dense {
	dX, dWq, dWk, dWv, dWout := attn.BackwardGradsOnly(dY)

	attn.t++
	lr := attn.learningRate
	for h := 0; h < attn.H; h++ {
		adamUpdateInPlace(attn.Wquery[h], dWq[h], attn.mWq[h], attn.vWq[h], attn.t,
			lr, config.AdamBeta1, config.AdamBeta2, config.AdamEps)
		adamUpdateInPlace(attn.Wkey[h], dWk[h], attn.mWk[h], attn.vWk[h], attn.t,
			lr, config.AdamBeta1, config.AdamBeta2, config.AdamEps)
		adamUpdateInPlace(attn.Wvalue[h], dWv[h], attn.mWv[h], attn.vWv[h], attn.t,
			lr, config.AdamBeta1, config.AdamBeta2, config.AdamEps)
	}
	adamUpdateInPlace(attn.Woutput, dWout, attn.mWo, attn.vWo, attn.t,
		lr, config.AdamBeta1, config.AdamBeta2, config.AdamEps)

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

	// Expand dY to full sequence if only the last position gradient was provided.
	dYr, dYc := dY.Dims()
	_, T := attn.X.Dims()
	if dYc == 1 && T > 1 {
		full := mat.NewDense(dYr, T, nil)
		for i := 0; i < dYr; i++ {
			full.Set(i, T-1, dY.At(i, 0))
		}
		dY = full
	}

	// dY with respect to Y = Wout * Ocat
	dWout = toDense(dot(dY, attn.O_cat.T()))
	dOcat := toDense(dot(attn.Woutput.T(), dY))

	dXtotal := mat.NewDense(attn.dModel, T, nil)

	row := 0
	rescale := 1.0 / math.Sqrt(float64(attn.dHead))

	for h := 0; h < attn.H; h++ {
		// slice out this head’s portion of dOcat
		dO := dOcat.Slice(row, row+attn.dHead, 0, T).(*mat.Dense)

		row += attn.dHead

		// O = V * A^T
		dV := toDense(dot(dO, attn.A[h]))       // (dHead x T)
		dA_T := toDense(dot(attn.V[h].T(), dO)) // (T x T)
		dA := dA_T.T()

		// A = softmax_row(S)
		dS := softmaxBackward(dA, attn.A[h]) // (T x T)

		// S = Q^T K / sqrt(dHead)
		dQ := toDense(scale(rescale, dot(attn.K[h], dS.T()))) // (dHead x T)
		dK := toDense(scale(rescale, dot(attn.Q[h], dS)))     // (dHead x T)

		// Params
		dWq[h] = toDense(dot(dQ, attn.X.T()))
		dWk[h] = toDense(dot(dK, attn.X.T()))
		dWv[h] = toDense(dot(dV, attn.X.T()))

		// Inputs
		dXq := toDense(dot(attn.Wquery[h].T(), dQ))
		dXk := toDense(dot(attn.Wkey[h].T(), dK))
		dXv := toDense(dot(attn.Wvalue[h].T(), dV))
		dXh := toDense(add(add(dXq, dXk), dXv))
		dXtotal = toDense(add(dXtotal, dXh))
	}
	return dXtotal, dWq, dWk, dWv, dWout
}

// -------- KV cache for inference (last-timestep only) --------

type AttnKV struct {
	K []*mat.Dense // per head: (dHead x t)
	V []*mat.Dense // per head: (dHead x t)
	t int
}

func newAttnKV(H, dHead int) AttnKV {
	k := make([]*mat.Dense, H)
	v := make([]*mat.Dense, H)
	return AttnKV{K: k, V: v, t: 0}
}

// append column helper: returns a new matrix with one more column
func appendCol(dst, col *mat.Dense) *mat.Dense {
	r, c := 0, 0
	if dst != nil {
		r, c = dst.Dims()
	} else {
		r = col.RawMatrix().Rows
	}
	if col.RawMatrix().Cols != 1 {
		panic("appendCol expects (r x 1) column")
	}
	out := mat.NewDense(r, c+1, nil)
	// copy old
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out.Set(i, j, dst.At(i, j))
		}
		out.Set(i, c, col.At(i, 0))
	}
	return out
}

// ForwardLastWithKV computes only the last timestep output using cached K,V.
// xLast: (dModel x 1), returns yLast: (dModel x 1). Updates kv in-place.
func (attn *Attention) ForwardLastWithKV(xLast *mat.Dense, kv *AttnKV) *mat.Dense {
	if kv.K == nil || len(kv.K) != attn.H {
		*kv = newAttnKV(attn.H, attn.dHead)
	}
	rescale := 1.0 / math.Sqrt(float64(attn.dHead))
	headsCatLast := mat.NewDense(attn.dModel, 1, nil)
	// per-head local buffers
	for h := 0; h < attn.H; h++ {
		// q,k,v for last token
		var q, k, v mat.Dense
		q.Mul(attn.Wquery[h], xLast) // (dHead x 1)
		k.Mul(attn.Wkey[h], xLast)
		v.Mul(attn.Wvalue[h], xLast)
		// append to cache
		kv.K[h] = appendCol(kv.K[h], toDense(&k))
		kv.V[h] = appendCol(kv.V[h], toDense(&v))

		// cap cache length to SeqLen by dropping oldest columns
		if config.SeqLen > 0 {
			if cols := kv.K[h].RawMatrix().Cols; cols > config.SeqLen {
				start := cols - config.SeqLen
				kv.K[h] = kv.K[h].Slice(0, kv.K[h].RawMatrix().Rows, start, cols).(*mat.Dense)
				kv.V[h] = kv.V[h].Slice(0, kv.V[h].RawMatrix().Rows, start, cols).(*mat.Dense)
			}
		}

		// scores for last row: (1 x T)
		var s mat.Dense
		s.Mul(q.T(), kv.K[h])
		s.Scale(rescale, &s)
		// softmax of row vector (1 x T)
		// Reuse RowSoftmax on a 1-row matrix by wrapping
		Arow := RowSoftmax(toDense(&s))
		// O_last = V * Arow^T  => (dHead x 1)
		var o mat.Dense
		o.Mul(kv.V[h], Arow.T())
		// write into headsCatLast rows
		base := h * attn.dHead
		dst := headsCatLast.Slice(base, base+attn.dHead, 0, 1).(*mat.Dense)
		dst.Copy(toDense(&o))
	}
	// update cached length
	if kv.K[0] != nil {
		kv.t = kv.K[0].RawMatrix().Cols
	} else {
		kv.t = 0
	}
	// output projection
	var yLast mat.Dense
	yLast.Mul(attn.Woutput, headsCatLast)
	return toDense(&yLast)
}
