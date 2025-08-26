package main

import (
	"fmt"
	"math/rand"
)

// Keep transformer.go unchanged. We define layers here
// because transformer.go references it.
var layers = 2

func main() {
	// Deterministic
	rand.Seed(42)

	// Square model so residual adds are valid
	d := 8
	attnLR := 0.10
	mlpLR := 0.10
	gpt := CreateGPT(d, d, d, attnLR, mlpLR)

	x := vector([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8})
	target := oneHot(d, 3)

	// First forward
	logits := forwardThrough(gpt, x)
	loss0, grad := CrossEntropyWithGrad(logits, target)
	fmt.Printf("Initial loss: %.6f\n", loss0)

	// Inspect attention weights of block 0 (row sums should be ~1)
	if A := gpt.blocks[0].attn.A; A != nil {
		fmt.Printf("Block 0 attention row sums: %v\n", rowSums(A))
	}

	// One backward step through all layers
	for i := layers - 1; i >= 0; i-- {
		grad = gpt.blocks[i].Backward(grad)
	}

	// Forward again
	logits2 := forwardThrough(gpt, x)
	loss1, _ := CrossEntropyWithGrad(logits2, target)
	fmt.Printf("Loss after 1 step: %.6f\n", loss1)
	fmt.Printf("Loss decreased: %v\n", loss1 < loss0)
	fmt.Printf("Logits (first run) head: %v\n", headVec(logits, 5))
	fmt.Printf("Logits (after 1 step) head: %v\n", headVec(logits2, 5))
}