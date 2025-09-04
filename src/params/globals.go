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
	AttnLR     float64
	MLPLR      float64
	UnembedLR  float64

	// Optimization/training wheel parameters
	NormLR      float64
	WarmupSteps int     // linear warmup steps
	DecaySteps  int     // cosine decay steps after warmup (0 = none)
	AdamBeta1   float64 // default 0.9
	AdamBeta2   float64 // default 0.999
	AdamEps     float64 // default 1e-8

	MaxEpochs int     // maximum number of epochs
	Patience  int     // early stopping patience
	Epsilon   float64 // stop if loss < epsilon
	BatchSize int     // mini-batch size
	ValFrac   float64 // fraction of data held out for validation

	// Stability parameters
	GradClip    float64 // <=0 disables (default 1.0 is a good start)
	WeightDecay float64 // AdamW-style, e.g., 0.01; 0 disables
	Debug       bool    // enable periodic debug logs
	DebugEvery  int     // print every N optimizer steps
	PosLR       float64 // learning rate for positional embeddings
    SaveEverySteps int  // checkpoint every N optimizer steps (0=disable)
}


// How many times does attn --> mlp happen
var Layers = 6
var Config = TrainingConfig{
	DModel:     512,
	HiddenSize: 1024,
	VocabSize:  8192, // Top number of 1-4 chars
	NumHeads:   8,    // dHead = DModel/NumHeads
	SeqLen:     64,   // max context
	AttnLR:     0.0003,
	MLPLR:      0.0003,
	UnembedLR:  0.00003,
	NormLR:     0.0003,

	MaxEpochs: 250,
	Patience:  25,
	Epsilon:   1e-4,
	BatchSize: 2048, // each example is one prefix
	ValFrac:   0.1,

	WarmupSteps: 10_000,
	DecaySteps:  1_000_000,
	AdamBeta1:   0.9,
	AdamBeta2:   0.999,
	AdamEps:     1e-8,

	GradClip:    1.0,
	WeightDecay: 0.01,
	Debug:       false,
	DebugEvery:  1000,
	PosLR:       0.0003,
    SaveEverySteps: 10000,
}
