package main

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)
type posData struct { R, C int; Data []float64 }

// Predict generates text autoregressively from an English prompt.
// It tokenizes the input, embeds it, runs it through the Transformer,
// and predicts next tokens until <eos> or maxLen.
func (gpt *Transformer) Predict(input string, maxLen int) []string {
	if emb == nil {
		panic("embeddings not initialized; call loadTrainingSet first")
	}

	// make sure <eos> is present at inference time
	ensureEOSToken()
	// Tokenize into 1–4 char pieces
	toks := tokenizeENPieces(input)
	// Start with <bos> + prompt
	toks = append(toks, "<eos>")
	seq := append([]string{"<bos>"}, toks...)
	ids := make([]int, len(seq))
	for i, t := range seq {
		ids[i] = vocabLookup(vocab, t)
	}
	// Per-block KV caches
	type blkKV struct{ attnKV AttnKV }
	kvs := make([]blkKV, layers)
	// Prime caches by rolling through the prompt tokens
	var yLast *mat.Dense
	for t := 0; t < len(ids); t++ {
		xLast := colAsVector(emb, ids[t])
		yLast = addPosCol(xLast, kvs[0].attnKV.t)
		for l := 0; l < layers; l++ {
			yLast = gpt.blocks[l].ForwardLastWithKV(yLast, &kvs[l].attnKV)
			// For blocks >0, do NOT add positional embedding again.
		}
	}
	// Generate up to maxLen new tokens
	for steps := 0; steps < maxLen; steps++ {
		logits := Unembed(yLast)
		probs := ColVectorSoftmax(logits)
		nextID := sampleFromProbs(probs, 50, 0.9)
		nextTok := vocab.IDToToken[nextID]
		if nextTok == "<eos>" {
			break
		}
		ids = append(ids, nextID)
		seq = append(seq, nextTok)
		// advance one step using KV cache
		xLast := colAsVector(emb, nextID)
		yLast = addPosCol(xLast, kvs[0].attnKV.t)
		for l := 0; l < layers; l++ {
			yLast = gpt.blocks[l].ForwardLastWithKV(yLast, &kvs[l].attnKV)
		}
	}
	// return generated tokens after the prompt
	fmt.Println("Input tokens:", toks)
	fmt.Println("Mapped IDs:", ids)
	return seq[1+len(toks):]
}

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

func SaveTransformer(gpt *Transformer, filename string) error {
	data := modelData{}
	data.Layers = len(gpt.blocks)
	data.Blocks = make([]blockData, len(gpt.blocks))

	for i, b := range gpt.blocks {
		attn := b.attn
		mlp := b.mlp

		// Heads
		hArr := make([]headData, attn.H)
		for h := 0; h < attn.H; h++ {
			// Q
			if attn.Wquery[h] != nil {
				r, c := attn.Wquery[h].Dims()
				raw := mat.DenseCopyOf(attn.Wquery[h]).RawMatrix()
				hArr[h].WqR, hArr[h].WqC = r, c
				hArr[h].WqData = append([]float64(nil), raw.Data...)
				if attn.mWq[h] != nil {
					mRaw := mat.DenseCopyOf(attn.mWq[h]).RawMatrix()
					vRaw := mat.DenseCopyOf(attn.vWq[h]).RawMatrix()
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
				if attn.mWk[h] != nil {
					mRaw := mat.DenseCopyOf(attn.mWk[h]).RawMatrix()
					vRaw := mat.DenseCopyOf(attn.vWk[h]).RawMatrix()
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
				if attn.mWv[h] != nil {
					mRaw := mat.DenseCopyOf(attn.mWv[h]).RawMatrix()
					vRaw := mat.DenseCopyOf(attn.vWv[h]).RawMatrix()
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
			if attn.mWo != nil {
				mRaw := mat.DenseCopyOf(attn.mWo).RawMatrix()
				vRaw := mat.DenseCopyOf(attn.vWo).RawMatrix()
				data.Blocks[i].MWo = append([]float64(nil), mRaw.Data...)
				data.Blocks[i].VWo = append([]float64(nil), vRaw.Data...)
			}
		}

		// MLP
		if mlp != nil {
			if mlp.hiddenWeights != nil {
				r, c := mlp.hiddenWeights.Dims()
				raw := mat.DenseCopyOf(mlp.hiddenWeights).RawMatrix()
				data.Blocks[i].HiddenWR, data.Blocks[i].HiddenWC = r, c
				data.Blocks[i].HiddenW = append([]float64(nil), raw.Data...)
				if mlp.mHiddenW != nil {
					mRaw := mat.DenseCopyOf(mlp.mHiddenW).RawMatrix()
					vRaw := mat.DenseCopyOf(mlp.vHiddenW).RawMatrix()
					data.Blocks[i].MHiddenW = append([]float64(nil), mRaw.Data...)
					data.Blocks[i].VHiddenW = append([]float64(nil), vRaw.Data...)
				}
			}
			if mlp.hiddenBias != nil {
				r, c := mlp.hiddenBias.Dims()
				raw := mat.DenseCopyOf(mlp.hiddenBias).RawMatrix()
				data.Blocks[i].HiddenBR, data.Blocks[i].HiddenBC = r, c
				data.Blocks[i].HiddenB = append([]float64(nil), raw.Data...)
				if mlp.mHiddenB != nil {
					mRaw := mat.DenseCopyOf(mlp.mHiddenB).RawMatrix()
					vRaw := mat.DenseCopyOf(mlp.vHiddenB).RawMatrix()
					data.Blocks[i].MHiddenB = append([]float64(nil), mRaw.Data...)
					data.Blocks[i].VHiddenB = append([]float64(nil), vRaw.Data...)
				}
			}
			if mlp.outputWeights != nil {
				r, c := mlp.outputWeights.Dims()
				raw := mat.DenseCopyOf(mlp.outputWeights).RawMatrix()
				data.Blocks[i].OutputWR, data.Blocks[i].OutputWC = r, c
				data.Blocks[i].OutputW = append([]float64(nil), raw.Data...)
				if mlp.mOutputW != nil {
					mRaw := mat.DenseCopyOf(mlp.mOutputW).RawMatrix()
					vRaw := mat.DenseCopyOf(mlp.vOutputW).RawMatrix()
					data.Blocks[i].MOutputW = append([]float64(nil), mRaw.Data...)
					data.Blocks[i].VOutputW = append([]float64(nil), vRaw.Data...)
				}
			}
			if mlp.outputBias != nil {
				r, c := mlp.outputBias.Dims()
				raw := mat.DenseCopyOf(mlp.outputBias).RawMatrix()
				data.Blocks[i].OutputBR, data.Blocks[i].OutputBC = r, c
				data.Blocks[i].OutputB = append([]float64(nil), raw.Data...)
				if mlp.mOutputB != nil {
					mRaw := mat.DenseCopyOf(mlp.mOutputB).RawMatrix()
					vRaw := mat.DenseCopyOf(mlp.vOutputB).RawMatrix()
					data.Blocks[i].MOutputB = append([]float64(nil), mRaw.Data...)
					data.Blocks[i].VOutputB = append([]float64(nil), vRaw.Data...)
				}
			}
		}

		// LayerNorms
		if b.ln1 != nil {
			gRaw := mat.DenseCopyOf(b.ln1.gamma).RawMatrix()
			bRaw := mat.DenseCopyOf(b.ln1.beta).RawMatrix()
			data.Blocks[i].Ln1Gamma = append([]float64(nil), gRaw.Data...)
			data.Blocks[i].Ln1Beta  = append([]float64(nil), bRaw.Data...)

			mG := mat.DenseCopyOf(b.ln1.mGamma).RawMatrix()
			vG := mat.DenseCopyOf(b.ln1.vGamma).RawMatrix()
			mB := mat.DenseCopyOf(b.ln1.mBeta).RawMatrix()
			vB := mat.DenseCopyOf(b.ln1.vBeta).RawMatrix()
			data.Blocks[i].Ln1MGamma = append([]float64(nil), mG.Data...)
			data.Blocks[i].Ln1VGamma = append([]float64(nil), vG.Data...)
			data.Blocks[i].Ln1MBeta  = append([]float64(nil), mB.Data...)
			data.Blocks[i].Ln1VBeta  = append([]float64(nil), vB.Data...)
		}
		if b.ln2 != nil {
			gRaw := mat.DenseCopyOf(b.ln2.gamma).RawMatrix()
			bRaw := mat.DenseCopyOf(b.ln2.beta).RawMatrix()
			data.Blocks[i].Ln2Gamma = append([]float64(nil), gRaw.Data...)
			data.Blocks[i].Ln2Beta  = append([]float64(nil), bRaw.Data...)

			mG := mat.DenseCopyOf(b.ln2.mGamma).RawMatrix()
			vG := mat.DenseCopyOf(b.ln2.vGamma).RawMatrix()
			mB := mat.DenseCopyOf(b.ln2.mBeta).RawMatrix()
			vB := mat.DenseCopyOf(b.ln2.vBeta).RawMatrix()
			data.Blocks[i].Ln2MGamma = append([]float64(nil), mG.Data...)
			data.Blocks[i].Ln2VGamma = append([]float64(nil), vG.Data...)
			data.Blocks[i].Ln2MBeta  = append([]float64(nil), mB.Data...)
			data.Blocks[i].Ln2VBeta  = append([]float64(nil), vB.Data...)
		}
	}

	// Embeddings
	if emb != nil {
		r, c := emb.Dims()
		raw := mat.DenseCopyOf(emb).RawMatrix()
		data.EmbR, data.EmbC = r, c
		data.EmbData = append([]float64(nil), raw.Data...)
		if embM != nil {
			mRaw := mat.DenseCopyOf(embM).RawMatrix()
			vRaw := mat.DenseCopyOf(embV).RawMatrix()
			data.EmbM = append([]float64(nil), mRaw.Data...)
			data.EmbV = append([]float64(nil), vRaw.Data...)
			data.EmbT = embT
		}
	}

	// Positional embeddings
	if posEmb != nil {
		r, c := posEmb.Dims()
		raw := mat.DenseCopyOf(posEmb).RawMatrix()
		data.PosR, data.PosC = r, c
		data.PosData = append([]float64(nil), raw.Data...)
		if posM != nil {
			mRaw := mat.DenseCopyOf(posM).RawMatrix()
			vRaw := mat.DenseCopyOf(posV).RawMatrix()
			data.PosM = append([]float64(nil), mRaw.Data...)
			data.PosV = append([]float64(nil), vRaw.Data...)
			data.PosT = posT
		}
	}

	// Vocab
	if len(vocab.IDToToken) > 0 {
		data.Vocab = append([]string(nil), vocab.IDToToken...)
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

	if len(gpt.blocks) != data.Layers {
		return fmt.Errorf("LoadTransformer: layer mismatch (have %d, file %d)", len(gpt.blocks), data.Layers)
	}

	for i := range gpt.blocks {
		b := &gpt.blocks[i]
		attn := b.attn
		mlp := b.mlp
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
					attn.mWq[h] = mat.NewDense(hd.WqR, hd.WqC, hd.MWq)
					attn.vWq[h] = mat.NewDense(hd.WqR, hd.WqC, hd.VWq)
				}
			}
			if hd.WkR > 0 {
				attn.Wkey[h] = mat.NewDense(hd.WkR, hd.WkC, hd.WkData)
				if len(hd.MKq) > 0 {
					attn.mWk[h] = mat.NewDense(hd.WkR, hd.WkC, hd.MKq)
					attn.vWk[h] = mat.NewDense(hd.WkR, hd.WkC, hd.VKq)
				}
			}
			if hd.WvR > 0 {
				attn.Wvalue[h] = mat.NewDense(hd.WvR, hd.WvC, hd.WvData)
				if len(hd.MVq) > 0 {
					attn.mWv[h] = mat.NewDense(hd.WvR, hd.WvC, hd.MVq)
					attn.vWv[h] = mat.NewDense(hd.WvR, hd.WvC, hd.VVq)
				}
			}
		}

		// Woutput
		if bd.WoR > 0 {
			attn.Woutput = mat.NewDense(bd.WoR, bd.WoC, bd.WoData)
			if len(bd.MWo) > 0 {
				attn.mWo = mat.NewDense(bd.WoR, bd.WoC, bd.MWo)
				attn.vWo = mat.NewDense(bd.WoR, bd.WoC, bd.VWo)
			}
		}

		// MLP
		if mlp != nil {
			if bd.HiddenWR > 0 {
				mlp.hiddenWeights = mat.NewDense(bd.HiddenWR, bd.HiddenWC, bd.HiddenW)
				if len(bd.MHiddenW) > 0 {
					mlp.mHiddenW = mat.NewDense(bd.HiddenWR, bd.HiddenWC, bd.MHiddenW)
					mlp.vHiddenW = mat.NewDense(bd.HiddenWR, bd.HiddenWC, bd.VHiddenW)
				}
			}
			if bd.HiddenBR > 0 {
				mlp.hiddenBias = mat.NewDense(bd.HiddenBR, bd.HiddenBC, bd.HiddenB)
				if len(bd.MHiddenB) > 0 {
					mlp.mHiddenB = mat.NewDense(bd.HiddenBR, bd.HiddenBC, bd.MHiddenB)
					mlp.vHiddenB = mat.NewDense(bd.HiddenBR, bd.HiddenBC, bd.VHiddenB)
				}
			}
			if bd.OutputWR > 0 {
				mlp.outputWeights = mat.NewDense(bd.OutputWR, bd.OutputWC, bd.OutputW)
				if len(bd.MOutputW) > 0 {
					mlp.mOutputW = mat.NewDense(bd.OutputWR, bd.OutputWC, bd.MOutputW)
					mlp.vOutputW = mat.NewDense(bd.OutputWR, bd.OutputWC, bd.VOutputW)
				}
			}
			if bd.OutputBR > 0 {
				mlp.outputBias = mat.NewDense(bd.OutputBR, bd.OutputBC, bd.OutputB)
				if len(bd.MOutputB) > 0 {
					mlp.mOutputB = mat.NewDense(bd.OutputBR, bd.OutputBC, bd.MOutputB)
					mlp.vOutputB = mat.NewDense(bd.OutputBR, bd.OutputBC, bd.VOutputB)
				}
			}
		}

		// LayerNorms
		if len(bd.Ln1Gamma) > 0 {
			b.ln1.gamma = mat.NewDense(b.ln1.d, 1, bd.Ln1Gamma)
			b.ln1.beta  = mat.NewDense(b.ln1.d, 1, bd.Ln1Beta)
			if len(bd.Ln1MGamma) > 0 {
				b.ln1.mGamma = mat.NewDense(b.ln1.d, 1, bd.Ln1MGamma)
				b.ln1.vGamma = mat.NewDense(b.ln1.d, 1, bd.Ln1VGamma)
				b.ln1.mBeta  = mat.NewDense(b.ln1.d, 1, bd.Ln1MBeta)
				b.ln1.vBeta  = mat.NewDense(b.ln1.d, 1, bd.Ln1VBeta)
			}
		}
		if len(bd.Ln2Gamma) > 0 {
			b.ln2.gamma = mat.NewDense(b.ln2.d, 1, bd.Ln2Gamma)
			b.ln2.beta  = mat.NewDense(b.ln2.d, 1, bd.Ln2Beta)
			if len(bd.Ln2MGamma) > 0 {
				b.ln2.mGamma = mat.NewDense(b.ln2.d, 1, bd.Ln2MGamma)
				b.ln2.vGamma = mat.NewDense(b.ln2.d, 1, bd.Ln2VGamma)
				b.ln2.mBeta  = mat.NewDense(b.ln2.d, 1, bd.Ln2MBeta)
				b.ln2.vBeta  = mat.NewDense(b.ln2.d, 1, bd.Ln2VBeta)
			}
		}
	}

	// Embeddings
	if data.EmbR > 0 && data.EmbC > 0 {
		emb = mat.NewDense(data.EmbR, data.EmbC, data.EmbData)
		if len(data.EmbM) > 0 {
			embM = mat.NewDense(data.EmbR, data.EmbC, data.EmbM)
			embV = mat.NewDense(data.EmbR, data.EmbC, data.EmbV)
			embT = data.EmbT
		}
	}

	// Positional embeddings
	if data.PosR > 0 && data.PosC > 0 {
		posEmb = mat.NewDense(data.PosR, data.PosC, data.PosData)
		if len(data.PosM) > 0 {
			posM = mat.NewDense(data.PosR, data.PosC, data.PosM)
			posV = mat.NewDense(data.PosR, data.PosC, data.PosV)
			posT = data.PosT
		}
	}

	// Vocab
	if len(data.Vocab) > 0 {
		vocab = Vocab{
			TokenToID: map[string]int{},
			IDToToken: append([]string(nil), data.Vocab...),
		}
		for i, tok := range vocab.IDToToken {
			vocab.TokenToID[tok] = i
		}
	}

	return nil
}
