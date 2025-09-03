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

type headData struct {
	WqData   []float64
	WqR, WqC int
	WkData   []float64
	WkR, WkC int
	WvData   []float64
	WvR, WvC int
}
type blockData struct {
	Heads             []headData
	WoData            []float64
	WoR, WoC          int
	HiddenW           []float64
	HiddenWR          int
	HiddenWC          int
	HiddenB           []float64
	HiddenBR          int
	HiddenBC          int
	OutputW           []float64
	OutputWR          int
	OutputWC          int
	OutputB           []float64
	OutputBR          int
	OutputBC          int
	Ln1Gamma, Ln1Beta []float64
	Ln2Gamma, Ln2Beta []float64
}

func SaveTransformer(gpt *Transformer, filename string) error {
	data := struct {
		Layers  int
		Blocks  []blockData
		EmbR    int
		EmbC    int
		EmbData []float64
		Pos     posData
		Vocab   []string
	}{}
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
			}
			// K
			if attn.Wkey[h] != nil {
				r, c := attn.Wkey[h].Dims()
				raw := mat.DenseCopyOf(attn.Wkey[h]).RawMatrix()
				hArr[h].WkR, hArr[h].WkC = r, c
				hArr[h].WkData = append([]float64(nil), raw.Data...)
			}
			// V
			if attn.Wvalue[h] != nil {
				r, c := attn.Wvalue[h].Dims()
				raw := mat.DenseCopyOf(attn.Wvalue[h]).RawMatrix()
				hArr[h].WvR, hArr[h].WvC = r, c
				hArr[h].WvData = append([]float64(nil), raw.Data...)
			}
		}
		data.Blocks[i].Heads = hArr
		// Woutput
		if attn.Woutput != nil {
			r, c := attn.Woutput.Dims()
			raw := mat.DenseCopyOf(attn.Woutput).RawMatrix()
			data.Blocks[i].WoR, data.Blocks[i].WoC = r, c
			data.Blocks[i].WoData = append([]float64(nil), raw.Data...)
		}
		// MLP
		if mlp != nil {
			// hidden weights/bias
			if mlp.hiddenWeights != nil {
				r, c := mlp.hiddenWeights.Dims()
				raw := mat.DenseCopyOf(mlp.hiddenWeights).RawMatrix()
				data.Blocks[i].HiddenWR, data.Blocks[i].HiddenWC = r, c
				data.Blocks[i].HiddenW = append([]float64(nil), raw.Data...)
			}

			// LayerNorms
			if b.ln1 != nil {
				gRaw := mat.DenseCopyOf(b.ln1.gamma).RawMatrix()
				bRaw := mat.DenseCopyOf(b.ln1.beta).RawMatrix()
				data.Blocks[i].Ln1Gamma = append([]float64(nil), gRaw.Data...)
				data.Blocks[i].Ln1Beta = append([]float64(nil), bRaw.Data...)
			}
			if b.ln2 != nil {
				gRaw := mat.DenseCopyOf(b.ln2.gamma).RawMatrix()
				bRaw := mat.DenseCopyOf(b.ln2.beta).RawMatrix()
				data.Blocks[i].Ln2Gamma = append([]float64(nil), gRaw.Data...)
				data.Blocks[i].Ln2Beta = append([]float64(nil), bRaw.Data...)
			}

			if mlp.hiddenBias != nil {
				r, c := mlp.hiddenBias.Dims()
				raw := mat.DenseCopyOf(mlp.hiddenBias).RawMatrix()
				data.Blocks[i].HiddenBR, data.Blocks[i].HiddenBC = r, c
				data.Blocks[i].HiddenB = append([]float64(nil), raw.Data...)
			}
			// output weights/bias
			if mlp.outputWeights != nil {
				r, c := mlp.outputWeights.Dims()
				raw := mat.DenseCopyOf(mlp.outputWeights).RawMatrix()
				data.Blocks[i].OutputWR, data.Blocks[i].OutputWC = r, c
				data.Blocks[i].OutputW = append([]float64(nil), raw.Data...)
			}
			if mlp.outputBias != nil {
				r, c := mlp.outputBias.Dims()
				raw := mat.DenseCopyOf(mlp.outputBias).RawMatrix()
				data.Blocks[i].OutputBR, data.Blocks[i].OutputBC = r, c
				data.Blocks[i].OutputB = append([]float64(nil), raw.Data...)
			}
		}
	}
	// Embeddings and vocab
	if emb != nil {
		r, c := emb.Dims()
		raw := mat.DenseCopyOf(emb).RawMatrix()
		data.EmbR, data.EmbC = r, c
		data.EmbData = append([]float64(nil), raw.Data...)
	}
	if len(vocab.IDToToken) > 0 {
		data.Vocab = append([]string(nil), vocab.IDToToken...)
	}
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(data); err != nil {
		return err
	}
	return os.WriteFile(filename, buf.Bytes(), 0644)
}

// LoadTransformer loads a Transformer saved by SaveTransformer into the provided gpt.
// It overwrites the blocks' attention and MLP weights. Also restores embeddings/vocab if present.
func LoadTransformer(gpt *Transformer, filename string) error {
	data := struct {
		Layers  int
		Blocks  []blockData
		EmbR    int
		EmbC    int
		EmbData []float64
		Pos     posData
		Vocab   []string
	}{}
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
		// Heads
		if len(data.Blocks[i].Heads) != attn.H {
			return fmt.Errorf("LoadTransformer: head count mismatch at block %d (have %d, file %d)", i, attn.H, len(data.Blocks[i].Heads))
		}
		for h := 0; h < attn.H; h++ {
			hd := data.Blocks[i].Heads[h]
			if hd.WqR > 0 {
				attn.Wquery[h] = mat.NewDense(hd.WqR, hd.WqC, hd.WqData)
			}
			if hd.WkR > 0 {
				attn.Wkey[h] = mat.NewDense(hd.WkR, hd.WkC, hd.WkData)
			}
			if hd.WvR > 0 {
				attn.Wvalue[h] = mat.NewDense(hd.WvR, hd.WvC, hd.WvData)
			}
		}
		// Woutput
		if data.Blocks[i].WoR > 0 {
			b.attn.Woutput = mat.NewDense(data.Blocks[i].WoR, data.Blocks[i].WoC, data.Blocks[i].WoData)
		}
		// MLP
		if mlp != nil {
			if data.Blocks[i].HiddenWR > 0 {
				mlp.hiddenWeights = mat.NewDense(data.Blocks[i].HiddenWR, data.Blocks[i].HiddenWC, data.Blocks[i].HiddenW)
			}
			if data.Blocks[i].HiddenBR > 0 {
				mlp.hiddenBias = mat.NewDense(data.Blocks[i].HiddenBR, data.Blocks[i].HiddenBC, data.Blocks[i].HiddenB)
			}
			if data.Blocks[i].OutputWR > 0 {
				mlp.outputWeights = mat.NewDense(data.Blocks[i].OutputWR, data.Blocks[i].OutputWC, data.Blocks[i].OutputW)
			}
			if data.Blocks[i].OutputBR > 0 {
				mlp.outputBias = mat.NewDense(data.Blocks[i].OutputBR, data.Blocks[i].OutputBC, data.Blocks[i].OutputB)
			}
		}
		// LayerNorm
		if len(data.Blocks[i].Ln1Gamma) > 0 {
			b.ln1.gamma = mat.NewDense(b.ln1.d, 1, data.Blocks[i].Ln1Gamma)
			b.ln1.beta = mat.NewDense(b.ln1.d, 1, data.Blocks[i].Ln1Beta)
		}
		if len(data.Blocks[i].Ln2Gamma) > 0 {
			b.ln2.gamma = mat.NewDense(b.ln2.d, 1, data.Blocks[i].Ln2Gamma)
			b.ln2.beta = mat.NewDense(b.ln2.d, 1, data.Blocks[i].Ln2Beta)
		}
	}
	// Restore embeddings and vocab if present
	if data.EmbR > 0 && data.EmbC > 0 && len(data.EmbData) == data.EmbR*data.EmbC {
		emb = mat.NewDense(data.EmbR, data.EmbC, data.EmbData)
	}

	 if data.Pos.R > 0 && data.Pos.C > 0 && len(data.Pos.Data) == data.Pos.R*data.Pos.C {
        posEmb = mat.NewDense(data.Pos.R, data.Pos.C, data.Pos.Data)
        // reset Adam state on load (optional)
        posM, posV = nil, nil
        posT = 0
    }

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
