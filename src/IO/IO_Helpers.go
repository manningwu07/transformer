package IO

import (
	"math/rand/v2"

	"github.com/manningwu07/GPT/params"
	"gonum.org/v1/gonum/mat"
)

// ------- EOS helpers (inference-time safety) --------
// Ensure <eos> exists in params.Vocab/emb. If added, append a small-random row in emb.
func EnsureEOSToken() {
    const eos = "<eos>"
    if params.Vocab.TokenToID == nil {
        return
    }
    if _, ok := params.Vocab.TokenToID[eos]; ok {
        return
    }
    id := len(params.Vocab.IDToToken)
    params.Vocab.TokenToID[eos] = id
    params.Vocab.IDToToken = append(params.Vocab.IDToToken, eos)
    if params.Emb != nil {
        r, c := params.Emb.Dims()
        if c == id { // grow by one COLUMN for the new token
            ne := mat.NewDense(r, c+1, nil)
            for i := 0; i < r; i++ {
                for j := 0; j < c; j++ {
                    ne.Set(i, j, params.Emb.At(i, j))
                }
                ne.Set(i, c, (rand.Float64()-0.5)*1e-3) // small random col
            }
            params.Emb = ne
        }
    }
}