package transformer

import (
	"math"
	"sync"

	"gonum.org/v1/gonum/mat"

	"github.com/manningwu07/GPT/optimizations"
	"github.com/manningwu07/GPT/params"
	"github.com/manningwu07/GPT/utils"
)

type Attention struct {
	H            int
	DModel       int
	DHead        int
	Wquery       []*mat.Dense
	Wkey         []*mat.Dense
	Wvalue       []*mat.Dense
	Woutput      *mat.Dense
	LearningRate float64

	// Adam
	T        int
	MWq, VWq []*mat.Dense
	MWk, VWk []*mat.Dense
	MWv, VWv []*mat.Dense
	MWo, VWo *mat.Dense

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
	headsCat := mat.NewDense(attn.DModel, T, nil)

	rescale := 1.0 / math.Sqrt(float64(attn.DHead))
	// cache mask by T
	mask, ok := attn.maskCache[T]
	if !ok {
		mask = utils.CausalMask(T)
		attn.maskCache[T] = mask
	}

	// prepare per-head scratch resized once per T
	if attn.lastT != T {
		for h := 0; h < attn.H; h++ {
			attn.Q[h] = mat.NewDense(attn.DHead, T, nil)
			attn.K[h] = mat.NewDense(attn.DHead, T, nil)
			attn.V[h] = mat.NewDense(attn.DHead, T, nil)
			attn.Scores[h] = mat.NewDense(T, T, nil)
			attn.O[h] = mat.NewDense(attn.DHead, T, nil)
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
		// Reuse buffer for A to avoid allocation each step
		if attn.A[h] == nil {
			attn.A[h] = mat.NewDense(T, T, nil)
		} else if ar, ac := attn.A[h].Dims(); ar != T || ac != T {
			attn.A[h] = mat.NewDense(T, T, nil)
		}
		utils.RowSoftmaxMaskedInPlace(attn.A[h], attn.Scores[h], mask)
		// O = V * A^T
		attn.O[h].Mul(attn.V[h], attn.A[h].T())
		// concat into headsCat rows
		base := h * attn.DHead
		dst := headsCat.Slice(base, base+attn.DHead, 0, T).(*mat.Dense)
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
		for h := 0; h < attn.H; h++ {
			work(h)
		}
	}
	attn.O_cat = headsCat
	// Debug: quick sanity check on head 0 attention row sums.
	if params.Config.Debug && attn.H > 0 && attn.T%params.Config.DebugEvery == 0 {
		a := attn.A[0]
		if a != nil {
			rs := utils.RowSums(a)
			mn, mx := rs[0], rs[0]
			for _, v := range rs {
				if v < mn {
					mn = v
				}
				if v > mx {
					mx = v
				}
			}
			utils.Debugf("Attn: head0 A row-sum min/max = %.4f/%.4f (T=%d)", mn, mx,
				len(rs))
		}
	}

	Y := utils.ToDense(utils.Dot(attn.Woutput, headsCat)) // (dModel x 1)
	return Y
}

// Backward: computes grads and updates weights (SGD)
func (attn *Attention) Backward(dY *mat.Dense) *mat.Dense {
	dX, dWq, dWk, dWv, dWout := attn.BackwardGradsOnly(dY)

	attn.T++
	lr := attn.LearningRate

	// Global per-module grad clipping (includes all heads + Wout)
	if params.Config.GradClip > 0 {
		// flatten slice for clip call
		grads := []*mat.Dense{dWout}
		for h := 0; h < attn.H; h++ {
			grads = append(grads, dWq[h], dWk[h], dWv[h])
		}
		s := utils.ClipGrads(params.Config.GradClip, grads...)
		if s < 1.0 && params.Config.Debug && attn.T%params.Config.DebugEvery == 0 {
			utils.Debugf("Attn: clipped grads by %.4f at step %d", s, attn.T)
		}
	}

	for h := 0; h < attn.H; h++ {
		optimizations.AdamUpdateInPlace(attn.Wquery[h], dWq[h], attn.MWq[h], attn.VWq[h],
			attn.T, lr, params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps,
			params.Config.WeightDecay)
		optimizations.AdamUpdateInPlace(attn.Wkey[h], dWk[h], attn.MWk[h], attn.VWk[h],
			attn.T, lr, params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps,
			params.Config.WeightDecay)
		optimizations.AdamUpdateInPlace(attn.Wvalue[h], dWv[h], attn.MWv[h], attn.VWv[h],
			attn.T, lr, params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps,
			params.Config.WeightDecay)
	}
	optimizations.AdamUpdateInPlace(attn.Woutput, dWout, attn.MWo, attn.VWo, attn.T, lr,
		params.Config.AdamBeta1, params.Config.AdamBeta2, params.Config.AdamEps, params.Config.WeightDecay)

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
	dWout = utils.ToDense(utils.Dot(dY, attn.O_cat.T()))
	dOcat := utils.ToDense(utils.Dot(attn.Woutput.T(), dY))

	dXtotal := mat.NewDense(attn.DModel, T, nil)

	row := 0
	rescale := 1.0 / math.Sqrt(float64(attn.DHead))

	for h := 0; h < attn.H; h++ {
		// slice out this head’s portion of dOcat
		dO := dOcat.Slice(row, row+attn.DHead, 0, T).(*mat.Dense)

		row += attn.DHead

		// O = V * A^T
		dV := utils.ToDense(utils.Dot(dO, attn.A[h]))       // (dHead x T)
		dA_T := utils.ToDense(utils.Dot(attn.V[h].T(), dO)) // (T x T)
		dA := dA_T.T()

		// A = softmax_row(S)
		dS := utils.SoftmaxBackward(dA, attn.A[h]) // (T x T)

		// S = Q^T K / sqrt(dHead)
		dQ := utils.ToDense(utils.Scale(rescale, utils.Dot(attn.K[h], dS.T()))) // (dHead x T)
		dK := utils.ToDense(utils.Scale(rescale, utils.Dot(attn.Q[h], dS)))     // (dHead x T)

		// Params
		dWq[h] = utils.ToDense(utils.Dot(dQ, attn.X.T()))
		dWk[h] = utils.ToDense(utils.Dot(dK, attn.X.T()))
		dWv[h] = utils.ToDense(utils.Dot(dV, attn.X.T()))

		// Inputs
		dXq := utils.ToDense(utils.Dot(attn.Wquery[h].T(), dQ))
		dXk := utils.ToDense(utils.Dot(attn.Wkey[h].T(), dK))
		dXv := utils.ToDense(utils.Dot(attn.Wvalue[h].T(), dV))
		dXh := utils.ToDense(utils.Add(utils.Add(dXq, dXk), dXv))
		dXtotal = utils.ToDense(utils.Add(dXtotal, dXh))
	}
	return dXtotal, dWq, dWk, dWv, dWout
}

// -------- KV cache for inference (last-timestep only) --------

type AttnKV struct {
	K []*mat.Dense // per head: (dHead x t)
	V []*mat.Dense // per head: (dHead x t)
	T int
}

func newAttnKV(H, dHead int) AttnKV {
	k := make([]*mat.Dense, H)
	v := make([]*mat.Dense, H)
	return AttnKV{K: k, V: v, T: 0}
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
		*kv = newAttnKV(attn.H, attn.DHead)
	}
	rescale := 1.0 / math.Sqrt(float64(attn.DHead))
	headsCatLast := mat.NewDense(attn.DModel, 1, nil)
	// per-head local buffers
	for h := 0; h < attn.H; h++ {
		// q,k,v for last token
		var q, k, v mat.Dense
		q.Mul(attn.Wquery[h], xLast) // (dHead x 1)
		k.Mul(attn.Wkey[h], xLast)
		v.Mul(attn.Wvalue[h], xLast)
		// append to cache
		kv.K[h] = appendCol(kv.K[h], utils.ToDense(&k))
		kv.V[h] = appendCol(kv.V[h], utils.ToDense(&v))

		// cap cache length to SeqLen by dropping oldest columns
		if params.Config.SeqLen > 0 {
			if cols := kv.K[h].RawMatrix().Cols; cols > params.Config.SeqLen {
				start := cols - params.Config.SeqLen
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
		Arow := utils.RowSoftmax(utils.ToDense(&s))
		// O_last = V * Arow^T  => (dHead x 1)
		var o mat.Dense
		o.Mul(kv.V[h], Arow.T())
		// write into headsCatLast rows
		base := h * attn.DHead
		dst := headsCatLast.Slice(base, base+attn.DHead, 0, 1).(*mat.Dense)
		dst.Copy(utils.ToDense(&o))
	}
	// update cached length
	if kv.K[0] != nil {
		kv.T = kv.K[0].RawMatrix().Cols
	} else {
		kv.T = 0
	}
	// output projection
	var yLast mat.Dense
	yLast.Mul(attn.Woutput, headsCatLast)
	return utils.ToDense(&yLast)
}
