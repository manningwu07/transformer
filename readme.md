# ChatGPT2-super mini (57.5M params) - in GoLang

This project implements a GPT-like transformer model for text generation, written entirely in Go. It supports training from scratch, resuming from checkpoints, and interactive text generation via a CLI. The design optionally integrats Apple's Accelerate framework for performance on compatible hardware.

This repository is completely meant for educational purposes. Please properly cite the author (see #Citation)

## Features

*   **Transformer Architecture:** Implements a multi-layer transformer decoder, including multi-head self-attention, GELU activation, and Layer Normalization.
*   **Tokenization:** Simple ASCII-only piece-based tokenizer.
*   **Training:**
    *   Streaming dataset processing to handle large text files without loading all into memory.
    *   AdamW optimizer with linear warmup and cosine decay learning rate schedule.
    *   Gradient clipping and weight decay for stability.
    *   Checkpointing: Automatic saving during graceful exits (`Ctrl+C`), manual trigger (`SIGUSR1`), and periodic epoch-based saves.
    *   Early stopping based on validation metrics.
    *   "Tiny Overfit" mode for quick sanity checks of the training loop.
*   **Inference:** Autoregressive text generation with KV caching for efficient decoding.
*   **Command-Line Interface:** Simple chat interface for interacting with the trained model.
*   **Accelerate Integration:** Optional (but recommended) use of Apple's Accelerate framework for BLAS operations on macOS, providing significant speedups.

## 1. Getting Started

### Prerequisites

*   Go (version 1.22 or higher recommended)
*   Decently powerful computer (M4 Pro or similar computer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/manningwu07/GPT.git
    cd GPT
    ```
2.  **Download Go modules:**
    ```bash
    go mod download
    ```
3.  **Build the project (optional, but good for testing):**
    ```bash
    go build -tags accelerate . -o mygpt
    ./mygpt
    ```

    or

     ```bash
    go run -tags accelerate .
    ```
    (Remove `-tags accelerate` if not on macOS or if you encounter issues)

### Data Preparation

The model expects training and evaluation data in plain text files.

1.  **Create a `data` directory:**
    ```bash
    mkdir -p data/raw
    ```
2.  **Place your training data:**
    *   The primary training file should be named `train.en` (or `train.eng`) within `data/raw/` or anywhere under the `data/` directory.
    *   Example: `data/raw/train.en`
3.  **Place your evaluation data (optional, but recommended):**
    *   The evaluation file should be named `eval.en` (or similar, containing "eval" and ending with `.en`) within `data/` subdirectories.
    *   Example: `data/test/eval.en`

The `IO` package will attempt to locate these files. If you place them elsewhere, you might need to adjust the `IO.findEnglishFile()` and `IO.findEvalFile()` functions.

## 2. Configuration

Model hyperparameters and training parameters are defined in `params/params.go`. You can adjust these to fit your dataset and computational resources.

```go
// params/params.go

var Config = TrainingConfig{
	DModel:     768,       // Model width
	HiddenSize: 2048,      // MLP hidden layer size
	VocabSize:  16384,     // Maximum vocabulary size (top 1-4 char pieces)
	NumHeads:   12,        // Number of attention heads (dHead = DModel/NumHeads)
	SeqLen:     1024,      // Maximum context length for training and inference
	AttnLR:     0.0003,    // Learning rate for attention weights
	MLPLR:      0.0003,    // Learning rate for MLP weights
	UnembedLR:  0.00003,   // Learning rate for the embedding layer during unembedding
	NormLR:     0.0003,    // Learning rate for Layer Normalization parameters

	MaxEpochs: 100,
	Patience:  15,         // Early stopping patience (epochs without improvement)
	SaveEpochNumber: 10,   // Save a checkpoint every N epochs
	ImprovementThreshold: 0.005, // Minimum accuracy improvement to save best model
	Epsilon:   1e-4,       // Stop if average loss falls below this
	BatchSize: 4096,       // Mini-batch size (number of sequences sampled per step)
	ValFrac:   0.1,        // Fraction of data held out for validation (currently not used explicitly, eval.en is separate)

	WarmupSteps: 10_000,   // Linear learning rate warmup steps
	DecaySteps:  1_000_000, // Cosine decay steps after warmup
	AdamBeta1:   0.9,      // Adam optimizer beta1
	AdamBeta2:   0.999,    // Adam optimizer beta2
	AdamEps:     1e-8,     // Adam optimizer epsilon

	GradClip:    1.0,      // Gradient clipping threshold (0 disables)
	WeightDecay: 0.01,     // AdamW-style weight decay (0 disables)
	Debug:       false,    // Enable debug logging
	DebugEvery:  1000,     // Print debug logs every N optimizer steps
	PosLR:       0.0003,   // Learning rate for positional embeddings
    SaveEverySteps: 10000, // Checkpoint every N optimizer steps (0=disable, currently not fully implemented)
}

var Layers = 8 // Number of transformer blocks
```

## 3. Training the Model

Training can be initiated via `go run .` or by running the compiled executable.

### Starting New Training

```bash
go run . -tags accelerate
```

*   **`--tags accelerate`**: (macOS only) Links against Apple's Accelerate framework for optimized BLAS. Remove if not on macOS or if building errors occur.
*   **Output**: The console will show epoch-wise progress, including accuracy, token loss, training perplexity (PPL), and evaluation perplexity.
*   **Checkpoints**: Models will be saved in the `models/` directory:
    *   `models/ckpt_latest.gob`: Saved on graceful exit (`Ctrl+C`) or `SIGUSR1`.
    *   `models/epoch_NNN.gob`: Saved periodically every `SaveEpochNumber` epochs.
    *   `models/best_model.gob`: Saved when a new best evaluation accuracy is achieved.
    *   `models/transformer.gob`: Final best model saved at the end of training.

### Resuming Training

To continue training from a previously saved checkpoint:

```bash
go run . -tags accelerate -resume <desiredChecpoint>
```

### Tiny Overfit Mode

To quickly verify that the training loop and gradient calculations are working, you can enable "tiny overfit" mode. This trains on a very small subset of your data (the first `N` lines) for a fixed number of steps, aiming for a rapid reduction in loss.

```bash
OVERFIT_TINY=1 go run . -tags accelerate
```

This mode will print `tokLoss` and `ppl` values, which should decrease steadily. After completion, it enters the `ChatCLI` for testing.

## 4. Interacting with the Model

Once training is complete (or if you loaded a pre-trained model), the program will automatically enter a chat CLI. 
Type your prompt, and the model will generate a response. Type `exit` to quit.


# Educational Notes:

This project is intended for learning and experimentation, not for training large-scale GPT models.
Key points:

- Attention: implemented with causal masking and KV caching for inference.
- MLP: two-layer feedforward with GELU activation.
- LayerNorm: custom implementation with Adam updates.
- Embeddings: learned token + positional embeddings.
- Training loop: manual forward/backward passes, gradient checks, AdamW updates.
- Evaluation: computes token-level accuracy and perplexity on eval.en.

The code is structured to make each component (attention, MLP, layer norm, embeddings) explicit and inspectable.

# Limitations

- Not optimized for GPU training (CPU-only).
- Memory usage grows with model size and sequence length.
- Tokenization is simplistic (ASCII 1–4 char pieces).
- Intended for educational purposes, not production deployment.

# Citation

If you use this code for teaching, demos, or experiments, please credit:

```bash
Educational GPT implementation in Go
https://github.com/manningwu07/transformer
```