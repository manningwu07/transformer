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
	SaveEpochNumber int // Saves modal every X epoch (for safety)
	ImprovementThreshold float64 // How much should the model improve before being saved
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

// Background specs---Cosume 8/14 cores at peak height (run cmd)
// WORKERS=4 GOMAXPROCS=8 VECLIB_MAXIMUM_THREADS=1 go run .

// Overnight---Consume almost all the resources; 12/14 cores (run cmd)
// WORKERS=4 GOMAXPROCS=12 VECLIB_MAXIMUM_THREADS=3 go run .

// How many times does attn --> mlp happen
var Layers = 8
var Config = TrainingConfig{
	// These are fundemental for fine tuning (DO NOT CHANGE)
	DModel:     768, 
	HiddenSize: 2048, 
	VocabSize:  16384, // Top number of 1-4 chars
	NumHeads:   12,   // dHead = DModel/NumHeads

	SeqLen:     1024,   // context window (num in tokens)
	AttnLR:     0.0003,
	MLPLR:      0.0003,
	UnembedLR:  0.00003,
	NormLR:     0.0003,

	MaxEpochs: 1000,
	Patience:  15,
	SaveEpochNumber: 25,
	ImprovementThreshold: 0.005,
	Epsilon:   1e-4,
	BatchSize: 32, // each example is one prefix
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
