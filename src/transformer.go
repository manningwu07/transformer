package main

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)

// Predict takes an English input string, tokenizes it, embeds it,
// runs it through the Transformer, and returns the top-1 predicted
// Chinese tokens (as strings).
func (gpt *Transformer) Predict(input string, maxLen int) []string {
	if embEN == nil || embZH == nil {
		panic("embeddings not initialized; call loadTrainingSet first")
	}

	// Tokenize English input
	enTokens := tokenizeEN(input)
	if len(enTokens) == 0 {
		return []string{}
	}

	outputs := []string{}

	// For each input token, predict one Chinese token
	for _, tok := range enTokens {
		enID := vocabLookup(enVocab, tok)
		X := embed(embEN, enID)

		// Forward through all layers
		for l := 0; l < layers; l++ {
			X = gpt.blocks[l].Forward(X)
		}

		// Unembed into ZH vocab logits
		logits := UnembedZH(X)

		// Softmax -> probabilities
		probs := ColVectorSoftmax(logits)

		// Argmax
		predID := argmaxVec(probs)
		predTok := zhVocab.IDToToken[predID]
		outputs = append(outputs, predTok)

		if len(outputs) >= maxLen {
			break
		}
	}

	return outputs
}

// PrintMatrix prints a Gonum matrix in a compact form.
func PrintMatrix(m mat.Matrix, name string) {
	r, c := m.Dims()
	fmt.Printf("Matrix %s (%dx%d):\n", name, r, c)
	fa := mat.Formatted(m, mat.Prefix("  "), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

// RowSums returns per-row sums for a mat.Dense.
func RowSums(m *mat.Dense) []float64 {
	r, c := m.Dims()
	out := make([]float64, r)
	for i := 0; i < r; i++ {
		sum := 0.0
		for j := 0; j < c; j++ {
			sum += m.At(i, j)
		}
		out[i] = sum
	}
	return out
}

// SaveTransformer persists a Transformer to disk using gob.
// It serializes only the numeric weight data (not function pointers).
// filename should be a path to create/overwrite.
func SaveTransformer(gpt *Transformer, filename string) error {
	// Build a serializable container
	type blockData struct {
		WqData   []float64
		WqR, WqC int

		WkData   []float64
		WkR, WkC int

		WvData   []float64
		WvR, WvC int

		WoData   []float64
		WoR, WoC int
	}

	data := struct {
		Layers int
		Blocks []blockData
	}{}

	data.Layers = len(gpt.blocks)
	data.Blocks = make([]blockData, len(gpt.blocks))

	for i, b := range gpt.blocks {
		attn := b.attn
		for h := 0; h < attn.H; h++ {
			// Wquery
			if attn.Wquery != nil {
				r, c := attn.Wquery[h].Dims()
				data.Blocks[i].WqR = r
				data.Blocks[i].WqC = c
				raw := mat.DenseCopyOf(attn.Wquery[h]).RawMatrix()
				data.Blocks[i].WqData = make([]float64, len(raw.Data))
				copy(data.Blocks[i].WqData, raw.Data)
			}
			// Wkey
			if attn.Wkey != nil {
				r, c := attn.Wkey[h].Dims()
				data.Blocks[i].WkR = r
				data.Blocks[i].WkC = c
				raw := mat.DenseCopyOf(attn.Wkey[h]).RawMatrix()
				data.Blocks[i].WkData = make([]float64, len(raw.Data))
				copy(data.Blocks[i].WkData, raw.Data)
			}
			// Wvalue
			if attn.Wvalue != nil {
				r, c := attn.Wvalue[h].Dims()
				data.Blocks[i].WvR = r
				data.Blocks[i].WvC = c
				raw := mat.DenseCopyOf(attn.Wvalue[h]).RawMatrix()
				data.Blocks[i].WvData = make([]float64, len(raw.Data))
				copy(data.Blocks[i].WvData, raw.Data)
			}
			// Woutput
			if attn.Woutput != nil {
				r, c := attn.Woutput.Dims()
				data.Blocks[i].WoR = r
				data.Blocks[i].WoC = c
				raw := mat.DenseCopyOf(attn.Woutput).RawMatrix()
				data.Blocks[i].WoData = make([]float64, len(raw.Data))
				copy(data.Blocks[i].WoData, raw.Data)
			}
		}

	}

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	if err := enc.Encode(data); err != nil {
		return err
	}

	return os.WriteFile(filename, buf.Bytes(), 0644)
}

// LoadTransformer loads a Transformer saved by SaveTransformer into the provided gpt.
// It will overwrite the attention weight matrices on gpt.blocks (creates matrices if nil).
func LoadTransformer(gpt *Transformer, filename string) error {
	type headData struct {
		WqData   []float64
		WqR, WqC int
		WkData   []float64
		WkR, WkC int
		WvData   []float64
		WvR, WvC int
	}
	type blockData struct {
		Heads    []headData
		WoData   []float64
		WoR, WoC int
	}
	data := struct {
		Layers int
		Blocks []blockData
	}{}

	data.Layers = len(gpt.blocks)
	data.Blocks = make([]blockData, len(gpt.blocks))

	for i, b := range gpt.blocks {
		attn := b.attn
		hArr := make([]headData, attn.H)
		for h := 0; h < attn.H; h++ {
			// Q
			if attn.Wquery != nil && attn.Wquery[h] != nil {
				r, c := attn.Wquery[h].Dims()
				raw := mat.DenseCopyOf(attn.Wquery[h]).RawMatrix()
				hArr[h].WqR, hArr[h].WqC = r, c
				hArr[h].WqData = append([]float64(nil), raw.Data...)
			}
			// K
			if attn.Wkey != nil && attn.Wkey[h] != nil {
				r, c := attn.Wkey[h].Dims()
				raw := mat.DenseCopyOf(attn.Wkey[h]).RawMatrix()
				hArr[h].WkR, hArr[h].WkC = r, c
				hArr[h].WkData = append([]float64(nil), raw.Data...)
			}
			// V
			if attn.Wvalue != nil && attn.Wvalue[h] != nil {
				r, c := attn.Wvalue[h].Dims()
				raw := mat.DenseCopyOf(attn.Wvalue[h]).RawMatrix()
				hArr[h].WvR, hArr[h].WvC = r, c
				hArr[h].WvData = append([]float64(nil), raw.Data...)
			}
		}
		data.Blocks[i].Heads = hArr
		if attn.Woutput != nil {
			r, c := attn.Woutput.Dims()
			raw := mat.DenseCopyOf(attn.Woutput).RawMatrix()
			data.Blocks[i].WoR, data.Blocks[i].WoC = r, c
			data.Blocks[i].WoData = append([]float64(nil), raw.Data...)
		}
	}

	var buf bytes.Buffer
 	enc := gob.NewEncoder(&buf)
 	if err := enc.Encode(data); err != nil {
 		return err
 	}
 
 	return os.WriteFile(filename, buf.Bytes(), 0644)
}
