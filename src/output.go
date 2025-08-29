package main

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)

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
	rawBytes, err := os.ReadFile(filename)
	if err != nil {
		return err
	}
	buf := bytes.NewBuffer(rawBytes)
	dec := gob.NewDecoder(buf)

	// Per-head data
	type headData struct {
		WqData   []float64
		WqR, WqC int

		WkData   []float64
		WkR, WkC int

		WvData   []float64
		WvR, WvC int
	}

	// Per-block data
	type blockData struct {
		Heads   []headData
		WoData  []float64
		WoR, WoC int
	}

	data := struct {
		Layers int
		Blocks []blockData
	}{}

	if err := dec.Decode(&data); err != nil {
		return err
	}

	// If sizes mismatch, be conservative
	n := len(gpt.blocks)
	if data.Layers < n {
		n = data.Layers
	}

	for i := 0; i < n; i++ {
		b := &gpt.blocks[i]
		attn := b.attn

		// Resize slices if needed
		if len(attn.Wquery) != len(data.Blocks[i].Heads) {
			attn.Wquery = make([]*mat.Dense, len(data.Blocks[i].Heads))
			attn.Wkey   = make([]*mat.Dense, len(data.Blocks[i].Heads))
			attn.Wvalue = make([]*mat.Dense, len(data.Blocks[i].Heads))
		}

		// Load per-head weights
		for h, hd := range data.Blocks[i].Heads {
			if len(hd.WqData) > 0 {
				attn.Wquery[h] = mat.NewDense(hd.WqR, hd.WqC, hd.WqData)
			}
			if len(hd.WkData) > 0 {
				attn.Wkey[h] = mat.NewDense(hd.WkR, hd.WkC, hd.WkData)
			}
			if len(hd.WvData) > 0 {
				attn.Wvalue[h] = mat.NewDense(hd.WvR, hd.WvC, hd.WvData)
			}
		}

		// Load shared output projection
		if len(data.Blocks[i].WoData) > 0 {
			attn.Woutput = mat.NewDense(data.Blocks[i].WoR, data.Blocks[i].WoC, data.Blocks[i].WoData)
		}
	}

	return nil
}