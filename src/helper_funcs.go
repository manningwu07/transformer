package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// Guard functions
func chooseValidHeads(dModel, preferred int) int {
    if preferred <= 0 {
        return 1
    }
    if dModel%preferred == 0 { 
        return preferred
    }

    best := 1
    limit := preferred
    if limit > dModel {
        limit = dModel
    }
    for h := limit; h >= 1; h-- {
        if dModel%h == 0 {
            fmt.Printf("Warning: using %d heads instead of %d\n", h, preferred)
            best = h
            break
        }
    }
    return best
}

// Ensures grad has same T as forward pass
func expandGradToSeq(grad *mat.Dense, lastInput *mat.Dense) *mat.Dense {
	_, T := lastInput.Dims()
    gr, gc := grad.Dims()
    if gc == T {
        return grad
    }
    if gc == 1 && T > 1 {
        full := mat.NewDense(gr, T, nil)
        for i := 0; i < gr; i++ {
            full.Set(i, T-1, grad.At(i, 0))
        }
        return full
    }
    panic(fmt.Sprintf("expandGradToSeq: grad has %d cols, expected 1 or %d", gc, T))
}


func randomArray(size int, v float64) []float64 {
	min := -1.0 / math.Sqrt(v+1e-12)
	max := 1.0 / math.Sqrt(v+1e-12)
	out := make([]float64, size)
	for i := 0; i < size; i++ {
		out[i] = min + (max-min)*rand.Float64()
	}
	return out
}

// Helper functions

func oneHot(n, idx int) *mat.Dense {
	v := make([]float64, n)
	if idx >= 0 && idx < n {
		v[idx] = 1.0
	}
	return mat.NewDense(n, 1, v)
}

func toDense(m mat.Matrix) *mat.Dense {
	if d, ok := m.(*mat.Dense); ok {
		return d
	}
	return mat.DenseCopyOf(m)
}


func matrixNorm(m *mat.Dense) float64 {
    r, c := m.Dims()
    s := 0.0
    for i := 0; i < r; i++ {
        for j := 0; j < c; j++ {
            v := m.At(i, j)
            s += v * v
        }
    }
    return math.Sqrt(s)
}

// ChatCLI


func chatCLI(gpt *Transformer) {
    reader := bufio.NewReader(os.Stdin)
    fmt.Println("ChatGPT-like CLI. Type 'exit' to quit.")
    for {
        fmt.Print("You: ")
        input, _ := reader.ReadString('\n')
        input = strings.TrimSpace(input)
        if input == "exit" {
            break
        }
        // Generate up to 50 tokens
        out := gpt.Predict(input, 50)
        fmt.Println("Bot:", strings.Join(out, " "))
    }
}


// ------- LayerNorm --------

// ensureNorms lazily allocates LayerNorms if they are nil.
func (b *TransformerBlock) ensureNorms(d int) {
	if b.ln1 == nil {
		b.ln1 = NewLayerNorm(d, 1e-5, config.NormLR)
	}
	if b.ln2 == nil {
		b.ln2 = NewLayerNorm(d, 1e-5, config.NormLR)
	}
}

// ------- LR schedule: linear warmup, then cosine decay --------
func LRSchedule(step int, peak float64) float64 {
	if step <= 0 {
		return 0
	}
	wu := config.WarmupSteps
	dec := config.DecaySteps
	if wu > 0 && step < wu {
		return peak * float64(step) / float64(wu)
	}
	if dec > 0 {
		x := float64(step-wu) / float64(dec)
		if x > 1 {
			x = 1
		} else if x < 0 {
			x = 0
		}
		scale := 0.5 * (1 + math.Cos(math.Pi*x))
		return peak * scale
	}
	return peak
}

// ------- Adam optimizer (in-place) --------

func initEmbAdamIfNeeded() {
	if emb != nil && embM == nil {
		embM = zerosLike(emb)
		embV = zerosLike(emb)
		embT = 0
	}
}

// p -= lr * mhat / (sqrt(vhat)+eps) with bias correction.
func adamUpdateInPlace(
	p, g, m, v *mat.Dense,
	t int,
	lr, beta1, beta2, eps float64,
) {
	pr, pc := p.Dims()
	if gr, gc := g.Dims(); gr != pr || gc != pc {
		panic("adamUpdateInPlace: grad shape mismatch")
	}
	if mr, mc := m.Dims(); mr != pr || mc != pc {
		panic("adamUpdateInPlace: m shape mismatch")
	}
	if vr, vc := v.Dims(); vr != pr || vc != pc {
		panic("adamUpdateInPlace: v shape mismatch")
	}
	b1t := math.Pow(beta1, float64(t))
	b2t := math.Pow(beta2, float64(t))
	c1 := 1.0 / (1.0 - b1t)
	c2 := 1.0 / (1.0 - b2t)
	for i := 0; i < pr; i++ {
		for j := 0; j < pc; j++ {
			gij := g.At(i, j)
			mij := beta1*m.At(i, j) + (1.0-beta1)*gij
			vij := beta2*v.At(i, j) + (1.0-beta2)*gij*gij
			mhat := mij * c1
			vhat := vij * c2
			pij := p.At(i, j) - lr*mhat/(math.Sqrt(vhat)+eps)
			m.Set(i, j, mij)
			v.Set(i, j, vij)
			p.Set(i, j, pij)
		}
	}
}

func zerosLike(a *mat.Dense) *mat.Dense {
	r, c := a.Dims()
	return mat.NewDense(r, c, nil)
}

func onesLike(a *mat.Dense) *mat.Dense {
	r, c := a.Dims()
	out := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ { for j := 0; j < c; j++ { out.Set(i, j, 1) } }
	return out
}