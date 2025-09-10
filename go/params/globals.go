package params

import "gonum.org/v1/gonum/mat"

// Embed structs and globals
type Vocabulary struct {
	TokenToID map[string]int
	IDToToken []string
}

// Globals initialized on first loadTrainingSet call.
var (
	Vocab Vocabulary
	Emb   *mat.Dense // (dModel x |V|)
)

// Adam Optimizer global vars
var (
	EmbM, EmbV *mat.Dense
	EmbT       int
	PosEmb *mat.Dense
	PosM, PosV *mat.Dense
	PosT int
)

type TrainingConfig struct {
	// Core transformer parameters
	DModel     int // model width
	HiddenSize int // MLP hidden
	VocabSize  int // |V|
	NumHeads   int // attention heads
	SeqLen     int // max prefix length (context length)
}

// How many times does attn --> mlp happen
var Layers = 8
var Config = TrainingConfig{
	// These are fundemental for fine tuning (DO NOT CHANGE)
	DModel:     768, 
	HiddenSize: 2048, 
	VocabSize:  16384, // Top number of 1-4 chars
	NumHeads:   12,   // dHead = DModel/NumHeads
	SeqLen:     1024,   // context window (num in tokens)
}
