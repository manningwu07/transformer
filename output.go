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
		WqData []float64
		WqR, WqC int

		WkData []float64
		WkR, WkC int

		WvData []float64
		WvR, WvC int

		WoData []float64
		WoR, WoC int
	}

	data := struct {
		Layers int
		Blocks []blockData
	}{}

	data.Layers = len(gpt.blocks)
	data.Blocks = make([]blockData, len(gpt.blocks))

	for i, b := range gpt.blocks {
		// Wquery
		if b.attn.Wquery != nil {
			r, c := b.attn.Wquery.Dims()
			data.Blocks[i].WqR = r
			data.Blocks[i].WqC = c
			raw := mat.DenseCopyOf(b.attn.Wquery).RawMatrix()
			data.Blocks[i].WqData = make([]float64, len(raw.Data))
			copy(data.Blocks[i].WqData, raw.Data)
		}
		// Wkey
		if b.attn.Wkey != nil {
			r, c := b.attn.Wkey.Dims()
			data.Blocks[i].WkR = r
			data.Blocks[i].WkC = c
			raw := mat.DenseCopyOf(b.attn.Wkey).RawMatrix()
			data.Blocks[i].WkData = make([]float64, len(raw.Data))
			copy(data.Blocks[i].WkData, raw.Data)
		}
		// Wvalue
		if b.attn.Wvalue != nil {
			r, c := b.attn.Wvalue.Dims()
			data.Blocks[i].WvR = r
			data.Blocks[i].WvC = c
			raw := mat.DenseCopyOf(b.attn.Wvalue).RawMatrix()
			data.Blocks[i].WvData = make([]float64, len(raw.Data))
			copy(data.Blocks[i].WvData, raw.Data)
		}
		// Woutput
		if b.attn.Woutput != nil {
			r, c := b.attn.Woutput.Dims()
			data.Blocks[i].WoR = r
			data.Blocks[i].WoC = c
			raw := mat.DenseCopyOf(b.attn.Woutput).RawMatrix()
			data.Blocks[i].WoData = make([]float64, len(raw.Data))
			copy(data.Blocks[i].WoData, raw.Data)
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

	type blockData struct {
		WqData []float64
		WqR, WqC int

		WkData []float64
		WkR, WkC int

		WvData []float64
		WvR, WvC int

		WoData []float64
		WoR, WoC int
	}

	data := struct {
		Layers int
		Blocks []blockData
	}{}

	if err := dec.Decode(&data); err != nil {
		return err
	}

	// If sizes mismatch, be conservative: only load up to min(len(gpt.blocks), data.Layers)
	n := len(gpt.blocks)
	if data.Layers < n {
		n = data.Layers
	}

	for i := 0; i < n; i++ {
		b := &gpt.blocks[i]

		if len(data.Blocks[i].WqData) > 0 {
			b.attn.Wquery = mat.NewDense(data.Blocks[i].WqR, data.Blocks[i].WqC, data.Blocks[i].WqData)
		}
		if len(data.Blocks[i].WkData) > 0 {
			b.attn.Wkey = mat.NewDense(data.Blocks[i].WkR, data.Blocks[i].WkC, data.Blocks[i].WkData)
		}
		if len(data.Blocks[i].WvData) > 0 {
			b.attn.Wvalue = mat.NewDense(data.Blocks[i].WvR, data.Blocks[i].WvC, data.Blocks[i].WvData)
		}
		if len(data.Blocks[i].WoData) > 0 {
			b.attn.Woutput = mat.NewDense(data.Blocks[i].WoR, data.Blocks[i].WoC, data.Blocks[i].WoData)
		}
	}

	return nil
}