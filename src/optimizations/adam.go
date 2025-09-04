package optimizations

import (
	"math"

	"github.com/manningwu07/GPT/params"
	"gonum.org/v1/gonum/mat"
)

// p -= lr * (mhat/(sqrt(vhat)+eps) + wd * p) with bias correction (AdamW).
// p -= lr * mhat / (sqrt(vhat)+eps) with bias correction.
func AdamUpdateInPlace(
	p, g, m, v *mat.Dense,
	t int,
	lr, beta1, beta2, eps, weightDecay float64,
) {
	pr, pc := p.Dims()
	if gr, gc := g.Dims(); gr != pr || gc != pc {
		panic("adamUpdateInPlace: grad shape mismatch")
	}
	if mr, mc := m.Dims(); mr != pr || mc != pc {
		panic("adamUpdateInPlace: m shape mismatch")
	}
	if vr, vc := v.Dims(); vr != pr || vc != pc {
		panic("adamUpdateInPlace: v shape mismatch")
	}
	b1t := math.Pow(beta1, float64(t))
	b2t := math.Pow(beta2, float64(t))
	c1 := 1.0 / (1.0 - b1t)
	c2 := 1.0 / (1.0 - b2t)
	for i := 0; i < pr; i++ {
		for j := 0; j < pc; j++ {
			gij := g.At(i, j)
			mij := beta1*m.At(i, j) + (1.0-beta1)*gij
			vij := beta2*v.At(i, j) + (1.0-beta2)*gij*gij
			mhat := mij * c1
			vhat := vij * c2
			denom := math.Sqrt(vhat) + eps
            wdTerm := weightDecay * p.At(i, j)
            update := mhat/denom + wdTerm
            pij := p.At(i, j) - lr*update
			m.Set(i, j, mij)
			v.Set(i, j, vij)
			p.Set(i, j, pij)
		}
	}
}


// ------- Adam optimizer (in-place) --------

func InitPosAdamIfNeeded() {
    if params.PosEmb != nil && params.PosM == nil {
        params.PosM = zerosLike(params.PosEmb)
        params.PosV = zerosLike(params.PosEmb)
        params.PosT = 0
    }
}

func InitEmbAdamIfNeeded() {
	if params.Emb != nil && params.EmbM == nil {
		params.EmbM = zerosLike(params.Emb)
		params.EmbV = zerosLike(params.Emb)
		params.EmbT = 0
	}
}

func zerosLike(a *mat.Dense) *mat.Dense {
	r, c := a.Dims()
	return mat.NewDense(r, c, nil)
}
