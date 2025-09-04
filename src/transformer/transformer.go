package transformer

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"os"

	"github.com/manningwu07/GPT/params"
	"gonum.org/v1/gonum/mat"
)

// SaveTransformer persists a Transformer (weights only) to disk using gob.
// It serializes numeric weights for attention (all heads), output projection,
// and MLP (weights + biases). It also saves embeddings and vocab if available.
// filename should be a path to create/overwrite.


type modelData struct {
	Layers int
	Blocks []blockData

	// Embeddings
	EmbR, EmbC int
	EmbData []float64
	EmbM, EmbV []float64
	EmbT int

	// Positional embeddings
	PosR, PosC int
	PosData []float64
	PosM, PosV []float64
	PosT int

	// Vocab
	Vocab []string
}

// Attn only
type headData struct {
	WqData   []float64
	WqR, WqC int
	WkData   []float64
	WkR, WkC int
	WvData   []float64
	WvR, WvC int

	// Adam states
	MWq, VWq []float64
	MKq, VKq []float64
	MVq, VVq []float64
}

// Everything else
type blockData struct {
	Heads []headData

	// Attention output
	WoData []float64
	WoR, WoC int
	MWo, VWo []float64

	// MLP
	HiddenW []float64
	HiddenWR, HiddenWC int
	HiddenB []float64
	HiddenBR, HiddenBC int
	OutputW []float64
	OutputWR, OutputWC int
	OutputB []float64
	OutputBR, OutputBC int

	// Adam states for MLP
	MHiddenW, VHiddenW []float64
	MHiddenB, VHiddenB []float64
	MOutputW, VOutputW []float64
	MOutputB, VOutputB []float64

	// LayerNorm params + Adam
	Ln1Gamma, Ln1Beta []float64
	Ln2Gamma, Ln2Beta []float64
	Ln1MGamma, Ln1VGamma []float64
	Ln1MBeta, Ln1VBeta   []float64
	Ln2MGamma, Ln2VGamma []float64
	Ln2MBeta, Ln2VBeta   []float64
}

// save persists the whole Transformer using the gob-based SaveTransformer.
func Save(gpt Transformer) error {
	_ = os.MkdirAll("models", 0o755)
	path := "models/transformer.gob"
	return SaveTransformer(&gpt, path)
}


func SaveTransformer(gpt *Transformer, filename string) error {
	data := modelData{}
	data.Layers = len(gpt.Blocks)
	data.Blocks = make([]blockData, len(gpt.Blocks))

	for i, b := range gpt.Blocks {
		attn := b.Attn
		mlp := b.Mlp

		// Heads
		hArr := make([]headData, attn.H)
		for h := 0; h < attn.H; h++ {
			// Q
			if attn.Wquery[h] != nil {
				r, c := attn.Wquery[h].Dims()
				raw := mat.DenseCopyOf(attn.Wquery[h]).RawMatrix()
				hArr[h].WqR, hArr[h].WqC = r, c
				hArr[h].WqData = append([]float64(nil), raw.Data...)
				if attn.MWq[h] != nil {
					mRaw := mat.DenseCopyOf(attn.MWq[h]).RawMatrix()
					vRaw := mat.DenseCopyOf(attn.VWq[h]).RawMatrix()
					hArr[h].MWq = append([]float64(nil), mRaw.Data...)
					hArr[h].VWq = append([]float64(nil), vRaw.Data...)
				}
			}
			// K
			if attn.Wkey[h] != nil {
				r, c := attn.Wkey[h].Dims()
				raw := mat.DenseCopyOf(attn.Wkey[h]).RawMatrix()
				hArr[h].WkR, hArr[h].WkC = r, c
				hArr[h].WkData = append([]float64(nil), raw.Data...)
				if attn.MWk[h] != nil {
					mRaw := mat.DenseCopyOf(attn.MWk[h]).RawMatrix()
					vRaw := mat.DenseCopyOf(attn.VWk[h]).RawMatrix()
					hArr[h].MKq = append([]float64(nil), mRaw.Data...)
					hArr[h].VKq = append([]float64(nil), vRaw.Data...)
				}
			}
			// V
			if attn.Wvalue[h] != nil {
				r, c := attn.Wvalue[h].Dims()
				raw := mat.DenseCopyOf(attn.Wvalue[h]).RawMatrix()
				hArr[h].WvR, hArr[h].WvC = r, c
				hArr[h].WvData = append([]float64(nil), raw.Data...)
				if attn.MWv[h] != nil {
					mRaw := mat.DenseCopyOf(attn.MWv[h]).RawMatrix()
					vRaw := mat.DenseCopyOf(attn.VWv[h]).RawMatrix()
					hArr[h].MVq = append([]float64(nil), mRaw.Data...)
					hArr[h].VVq = append([]float64(nil), vRaw.Data...)
				}
			}
		}
		data.Blocks[i].Heads = hArr

		// Woutput
		if attn.Woutput != nil {
			r, c := attn.Woutput.Dims()
			raw := mat.DenseCopyOf(attn.Woutput).RawMatrix()
			data.Blocks[i].WoR, data.Blocks[i].WoC = r, c
			data.Blocks[i].WoData = append([]float64(nil), raw.Data...)
			if attn.MWo != nil {
				mRaw := mat.DenseCopyOf(attn.MWo).RawMatrix()
				vRaw := mat.DenseCopyOf(attn.VWo).RawMatrix()
				data.Blocks[i].MWo = append([]float64(nil), mRaw.Data...)
				data.Blocks[i].VWo = append([]float64(nil), vRaw.Data...)
			}
		}

		// MLP
		if mlp != nil {
			if mlp.HiddenWeights != nil {
				r, c := mlp.HiddenWeights.Dims()
				raw := mat.DenseCopyOf(mlp.HiddenWeights).RawMatrix()
				data.Blocks[i].HiddenWR, data.Blocks[i].HiddenWC = r, c
				data.Blocks[i].HiddenW = append([]float64(nil), raw.Data...)
				if mlp.MHiddenW != nil {
					mRaw := mat.DenseCopyOf(mlp.MHiddenW).RawMatrix()
					vRaw := mat.DenseCopyOf(mlp.VHiddenW).RawMatrix()
					data.Blocks[i].MHiddenW = append([]float64(nil), mRaw.Data...)
					data.Blocks[i].VHiddenW = append([]float64(nil), vRaw.Data...)
				}
			}
			if mlp.HiddenBias != nil {
				r, c := mlp.HiddenBias.Dims()
				raw := mat.DenseCopyOf(mlp.HiddenBias).RawMatrix()
				data.Blocks[i].HiddenBR, data.Blocks[i].HiddenBC = r, c
				data.Blocks[i].HiddenB = append([]float64(nil), raw.Data...)
				if mlp.MHiddenB != nil {
					mRaw := mat.DenseCopyOf(mlp.MHiddenB).RawMatrix()
					vRaw := mat.DenseCopyOf(mlp.VHiddenB).RawMatrix()
					data.Blocks[i].MHiddenB = append([]float64(nil), mRaw.Data...)
					data.Blocks[i].VHiddenB = append([]float64(nil), vRaw.Data...)
				}
			}
			if mlp.OutputWeights != nil {
				r, c := mlp.OutputWeights.Dims()
				raw := mat.DenseCopyOf(mlp.OutputWeights).RawMatrix()
				data.Blocks[i].OutputWR, data.Blocks[i].OutputWC = r, c
				data.Blocks[i].OutputW = append([]float64(nil), raw.Data...)
				if mlp.MOutputW != nil {
					mRaw := mat.DenseCopyOf(mlp.MOutputW).RawMatrix()
					vRaw := mat.DenseCopyOf(mlp.VOutputW).RawMatrix()
					data.Blocks[i].MOutputW = append([]float64(nil), mRaw.Data...)
					data.Blocks[i].VOutputW = append([]float64(nil), vRaw.Data...)
				}
			}
			if mlp.OutputBias != nil {
				r, c := mlp.OutputBias.Dims()
				raw := mat.DenseCopyOf(mlp.OutputBias).RawMatrix()
				data.Blocks[i].OutputBR, data.Blocks[i].OutputBC = r, c
				data.Blocks[i].OutputB = append([]float64(nil), raw.Data...)
				if mlp.MOutputB != nil {
					mRaw := mat.DenseCopyOf(mlp.MOutputB).RawMatrix()
					vRaw := mat.DenseCopyOf(mlp.VOutputB).RawMatrix()
					data.Blocks[i].MOutputB = append([]float64(nil), mRaw.Data...)
					data.Blocks[i].VOutputB = append([]float64(nil), vRaw.Data...)
				}
			}
		}

		// LayerNorms
		if b.Ln1 != nil {
			gRaw := mat.DenseCopyOf(b.Ln1.Gamma).RawMatrix()
			bRaw := mat.DenseCopyOf(b.Ln1.Beta).RawMatrix()
			data.Blocks[i].Ln1Gamma = append([]float64(nil), gRaw.Data...)
			data.Blocks[i].Ln1Beta  = append([]float64(nil), bRaw.Data...)

			mG := mat.DenseCopyOf(b.Ln1.MGamma).RawMatrix()
			vG := mat.DenseCopyOf(b.Ln1.VGamma).RawMatrix()
			mB := mat.DenseCopyOf(b.Ln1.MBeta).RawMatrix()
			vB := mat.DenseCopyOf(b.Ln1.VBeta).RawMatrix()
			data.Blocks[i].Ln1MGamma = append([]float64(nil), mG.Data...)
			data.Blocks[i].Ln1VGamma = append([]float64(nil), vG.Data...)
			data.Blocks[i].Ln1MBeta  = append([]float64(nil), mB.Data...)
			data.Blocks[i].Ln1VBeta  = append([]float64(nil), vB.Data...)
		}
		if b.Ln2 != nil {
			gRaw := mat.DenseCopyOf(b.Ln2.Gamma).RawMatrix()
			bRaw := mat.DenseCopyOf(b.Ln2.Beta).RawMatrix()
			data.Blocks[i].Ln2Gamma = append([]float64(nil), gRaw.Data...)
			data.Blocks[i].Ln2Beta  = append([]float64(nil), bRaw.Data...)

			mG := mat.DenseCopyOf(b.Ln2.MGamma).RawMatrix()
			vG := mat.DenseCopyOf(b.Ln2.VGamma).RawMatrix()
			mB := mat.DenseCopyOf(b.Ln2.MBeta).RawMatrix()
			vB := mat.DenseCopyOf(b.Ln2.VBeta).RawMatrix()
			data.Blocks[i].Ln2MGamma = append([]float64(nil), mG.Data...)
			data.Blocks[i].Ln2VGamma = append([]float64(nil), vG.Data...)
			data.Blocks[i].Ln2MBeta  = append([]float64(nil), mB.Data...)
			data.Blocks[i].Ln2VBeta  = append([]float64(nil), vB.Data...)
		}
	}

	// Embeddings
	if params.Emb != nil {
		r, c := params.Emb.Dims()
		raw := mat.DenseCopyOf(params.Emb).RawMatrix()
		data.EmbR, data.EmbC = r, c
		data.EmbData = append([]float64(nil), raw.Data...)
		if params.EmbM != nil {
			mRaw := mat.DenseCopyOf(params.EmbM).RawMatrix()
			vRaw := mat.DenseCopyOf(params.EmbV).RawMatrix()
			data.EmbM = append([]float64(nil), mRaw.Data...)
			data.EmbV = append([]float64(nil), vRaw.Data...)
			data.EmbT = params.EmbT
		}
	}

	// Positional embeddings
	if params.PosEmb != nil {
		r, c := params.PosEmb.Dims()
		raw := mat.DenseCopyOf(params.PosEmb).RawMatrix()
		data.PosR, data.PosC = r, c
		data.PosData = append([]float64(nil), raw.Data...)
		if params.PosM != nil {
			mRaw := mat.DenseCopyOf(params.PosM).RawMatrix()
			vRaw := mat.DenseCopyOf(params.PosV).RawMatrix()
			data.PosM = append([]float64(nil), mRaw.Data...)
			data.PosV = append([]float64(nil), vRaw.Data...)
			data.PosT = params.PosT
		}
	}

	// Vocab
	if len(params.Vocab.IDToToken) > 0 {
		data.Vocab = append([]string(nil), params.Vocab.IDToToken...)
	}

	// Encode
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(data); err != nil {
		return err
	}
	return os.WriteFile(filename, buf.Bytes(), 0644)
}

// LoadTransformer loads a Transformer saved by SaveTransformer into the provided gpt.
func LoadTransformer(gpt *Transformer, filename string) error {
	data := modelData{}

	// Read file and decode
	raw, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	dec := gob.NewDecoder(bytes.NewReader(raw))
	if err := dec.Decode(&data); err != nil {
		return err
	}

	if len(gpt.Blocks) != data.Layers {
		return fmt.Errorf("LoadTransformer: layer mismatch (have %d, file %d)", len(gpt.Blocks), data.Layers)
	}

	for i := range gpt.Blocks {
		b := &gpt.Blocks[i]
		attn := b.Attn
		mlp := b.Mlp
		bd := data.Blocks[i]

		// Heads
		if len(bd.Heads) != attn.H {
			return fmt.Errorf("LoadTransformer: head count mismatch at block %d (have %d, file %d)", i, attn.H, len(bd.Heads))
		}
		for h := 0; h < attn.H; h++ {
			hd := bd.Heads[h]
			if hd.WqR > 0 {
				attn.Wquery[h] = mat.NewDense(hd.WqR, hd.WqC, hd.WqData)
				if len(hd.MWq) > 0 {
					attn.MWq[h] = mat.NewDense(hd.WqR, hd.WqC, hd.MWq)
					attn.VWq[h] = mat.NewDense(hd.WqR, hd.WqC, hd.VWq)
				}
			}
			if hd.WkR > 0 {
				attn.Wkey[h] = mat.NewDense(hd.WkR, hd.WkC, hd.WkData)
				if len(hd.MKq) > 0 {
					attn.MWk[h] = mat.NewDense(hd.WkR, hd.WkC, hd.MKq)
					attn.VWk[h] = mat.NewDense(hd.WkR, hd.WkC, hd.VKq)
				}
			}
			if hd.WvR > 0 {
				attn.Wvalue[h] = mat.NewDense(hd.WvR, hd.WvC, hd.WvData)
				if len(hd.MVq) > 0 {
					attn.MWv[h] = mat.NewDense(hd.WvR, hd.WvC, hd.MVq)
					attn.VWv[h] = mat.NewDense(hd.WvR, hd.WvC, hd.VVq)
				}
			}
		}

		// Woutput
		if bd.WoR > 0 {
			attn.Woutput = mat.NewDense(bd.WoR, bd.WoC, bd.WoData)
			if len(bd.MWo) > 0 {
				attn.MWo = mat.NewDense(bd.WoR, bd.WoC, bd.MWo)
				attn.VWo = mat.NewDense(bd.WoR, bd.WoC, bd.VWo)
			}
		}

		// MLP
		if mlp != nil {
			if bd.HiddenWR > 0 {
				mlp.HiddenWeights = mat.NewDense(bd.HiddenWR, bd.HiddenWC, bd.HiddenW)
				if len(bd.MHiddenW) > 0 {
					mlp.MHiddenW = mat.NewDense(bd.HiddenWR, bd.HiddenWC, bd.MHiddenW)
					mlp.VHiddenW = mat.NewDense(bd.HiddenWR, bd.HiddenWC, bd.VHiddenW)
				}
			}
			if bd.HiddenBR > 0 {
				mlp.HiddenBias = mat.NewDense(bd.HiddenBR, bd.HiddenBC, bd.HiddenB)
				if len(bd.MHiddenB) > 0 {
					mlp.MHiddenB = mat.NewDense(bd.HiddenBR, bd.HiddenBC, bd.MHiddenB)
					mlp.VHiddenB = mat.NewDense(bd.HiddenBR, bd.HiddenBC, bd.VHiddenB)
				}
			}
			if bd.OutputWR > 0 {
				mlp.OutputWeights = mat.NewDense(bd.OutputWR, bd.OutputWC, bd.OutputW)
				if len(bd.MOutputW) > 0 {
					mlp.MOutputW = mat.NewDense(bd.OutputWR, bd.OutputWC, bd.MOutputW)
					mlp.VOutputW = mat.NewDense(bd.OutputWR, bd.OutputWC, bd.VOutputW)
				}
			}
			if bd.OutputBR > 0 {
				mlp.OutputBias = mat.NewDense(bd.OutputBR, bd.OutputBC, bd.OutputB)
				if len(bd.MOutputB) > 0 {
					mlp.MOutputB = mat.NewDense(bd.OutputBR, bd.OutputBC, bd.MOutputB)
					mlp.VOutputB = mat.NewDense(bd.OutputBR, bd.OutputBC, bd.VOutputB)
				}
			}
		}

		// LayerNorms
		if len(bd.Ln1Gamma) > 0 {
			b.Ln1.Gamma = mat.NewDense(b.Ln1.D, 1, bd.Ln1Gamma)
			b.Ln1.Beta  = mat.NewDense(b.Ln1.D, 1, bd.Ln1Beta)
			if len(bd.Ln1MGamma) > 0 {
				b.Ln1.MGamma = mat.NewDense(b.Ln1.D, 1, bd.Ln1MGamma)
				b.Ln1.VGamma = mat.NewDense(b.Ln1.D, 1, bd.Ln1VGamma)
				b.Ln1.MBeta  = mat.NewDense(b.Ln1.D, 1, bd.Ln1MBeta)
				b.Ln1.VBeta  = mat.NewDense(b.Ln1.D, 1, bd.Ln1VBeta)
			}
		}
		if len(bd.Ln2Gamma) > 0 {
			b.Ln2.Gamma = mat.NewDense(b.Ln2.D, 1, bd.Ln2Gamma)
			b.Ln2.Beta  = mat.NewDense(b.Ln2.D, 1, bd.Ln2Beta)
			if len(bd.Ln2MGamma) > 0 {
				b.Ln2.MGamma = mat.NewDense(b.Ln2.D, 1, bd.Ln2MGamma)
				b.Ln2.VGamma = mat.NewDense(b.Ln2.D, 1, bd.Ln2VGamma)
				b.Ln2.MBeta  = mat.NewDense(b.Ln2.D, 1, bd.Ln2MBeta)
				b.Ln2.VBeta  = mat.NewDense(b.Ln2.D, 1, bd.Ln2VBeta)
			}
		}
	}

	// Embeddings
	if data.EmbR > 0 && data.EmbC > 0 {
		params.Emb = mat.NewDense(data.EmbR, data.EmbC, data.EmbData)
		if len(data.EmbM) > 0 {
			params.EmbM = mat.NewDense(data.EmbR, data.EmbC, data.EmbM)
			params.EmbV = mat.NewDense(data.EmbR, data.EmbC, data.EmbV)
			params.EmbT = data.EmbT
		}
	}

	// Positional embeddings
	if data.PosR > 0 && data.PosC > 0 {
		params.PosEmb = mat.NewDense(data.PosR, data.PosC, data.PosData)
		if len(data.PosM) > 0 {
			params.PosM = mat.NewDense(data.PosR, data.PosC, data.PosM)
			params.PosV = mat.NewDense(data.PosR, data.PosC, data.PosV)
			params.PosT = data.PosT
		}
	}

	// Vocab
	if len(data.Vocab) > 0 {
		params.Vocab = params.Vocabulary{
			TokenToID: map[string]int{},
			IDToToken: append([]string(nil), data.Vocab...),
		}
		for i, tok := range params.Vocab.IDToToken {
			params.Vocab.TokenToID[tok] = i
		}
	}

	return nil
}
