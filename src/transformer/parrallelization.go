package transformer

import (
	"github.com/manningwu07/GPT/optimizations"
	"gonum.org/v1/gonum/mat"
)

// CloneForGradsOnly creates a shallow clone of the model where all weights/biases
// are shared (read-only), but per-module caches are private to avoid races.
// No optimizer state is copied. Safe for concurrent forwardBackwardGradsOnly*.
func (g *Transformer) CloneForGradsOnly() *Transformer {
	out := &Transformer{Blocks: make([]TransformerBlock, len(g.Blocks))}
	for i := range g.Blocks {
		src := &g.Blocks[i]
		dst := &out.Blocks[i]
		dst.Attn = cloneAttentionForGrads(src.Attn)
		dst.Mlp = cloneMLPForGrads(src.Mlp)
		dst.Ln1 = cloneLNForGrads(src.Ln1)
		dst.Ln2 = cloneLNForGrads(src.Ln2)
	}
	return out
}

func cloneAttentionForGrads(src *Attention) *Attention {
	a := &Attention{
		H:       src.H,
		DModel:  src.DModel,
		DHead:   src.DHead,
		Wquery:  src.Wquery, // shared read-only
		Wkey:    src.Wkey,   // shared read-only
		Wvalue:  src.Wvalue, // shared read-only
		Woutput: src.Woutput,
		// private caches
		Q:         make([]*mat.Dense, src.H),
		K:         make([]*mat.Dense, src.H),
		V:         make([]*mat.Dense, src.H),
		Scores:    make([]*mat.Dense, src.H),
		A:         make([]*mat.Dense, src.H),
		O:         make([]*mat.Dense, src.H),
		maskCache: make(map[int]*mat.Dense),
		lastT:     0,
		parallel:  false, // avoid head-level goroutines inside worker to reduce oversubscription
	}
	return a
}

func cloneMLPForGrads(src *MLP) *MLP {
	return &MLP{
		Inputs:        src.Inputs,
		Hiddens:       src.Hiddens,
		Outputs:       src.Outputs,
		HiddenWeights: src.HiddenWeights, // shared read-only
		HiddenBias:    src.HiddenBias,
		OutputWeights: src.OutputWeights,
		OutputBias:    src.OutputBias,
	}
}

func cloneLNForGrads(src *optimizations.LayerNorm) *optimizations.LayerNorm {
	ln := optimizations.NewLayerNorm(src.D, src.Eps, 0)
	ln.Gamma = src.Gamma // shared read-only
	ln.Beta = src.Beta
	// caches remain private per clone
	return ln
}