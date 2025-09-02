package main

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
)

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
         fmt.Println("Bot:", renderTokens(out))
    }
}

// renderTokens concatenates tokens and treats <eos> as newline.
func renderTokens(toks []string) string {
    if len(toks) == 0 {
        return ""
    }
    var sb strings.Builder
    for _, tk := range toks {
        if tk == "<eos>" {
            sb.WriteString("\n")
            break
        }
        sb.WriteString(tk)
    }
    return sb.String()
}


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


// ------- EOS helpers (inference-time safety) --------
// Ensure <eos> exists in vocab/emb. If added, append a small-random row in emb.
func ensureEOSToken() {
    const eos = "<eos>"
    if vocab.TokenToID == nil {
        return
    }
    if _, ok := vocab.TokenToID[eos]; ok {
        return
    }
    id := len(vocab.IDToToken)
    vocab.TokenToID[eos] = id
    vocab.IDToToken = append(vocab.IDToToken, eos)
    if emb != nil {
        r, c := emb.Dims()
        if r == id { // need to grow by 1
            ne := mat.NewDense(r+1, c, nil)
            ne.Slice(0, r, 0, c).(*mat.Dense).Copy(emb)
            // small random row
            for j := 0; j < c; j++ {
                ne.Set(r, j, (rand.Float64()-0.5)*1e-3)
            }
            emb = ne
        }
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

// Small helpers for debugging and clipping.
func matFroNorm(a *mat.Dense) float64 {
    r, c := a.Dims()
    s := 0.0
    for i := 0; i < r; i++ {
        for j := 0; j < c; j++ {
            v := a.At(i, j)
            s += v * v
        }
    }
    return math.Sqrt(s)
}

func scaleInPlace(a *mat.Dense, s float64) {
    if s == 1.0 {
        return
    }
    r, c := a.Dims()
    for i := 0; i < r; i++ {
        for j := 0; j < c; j++ {
            a.Set(i, j, a.At(i, j)*s)
        }
    }
}

// clipGrads scales all grads so their combined norm <= maxNorm.
// Returns the scale actually applied (<=1.0) or 1.0 if no clip.
func clipGrads(maxNorm float64, grads ...*mat.Dense) float64 {
    if maxNorm <= 0 {
        return 1.0
    }
    sum := 0.0
  	for _, g := range grads {
       if g == nil {
           continue
       }
       n := matFroNorm(g)
        sum += n * n
    }
    gn := math.Sqrt(sum)
    if gn <= maxNorm || gn == 0 {
        return 1.0
    }
    s := maxNorm / gn
    for _, g := range grads {
        if g != nil {
            scaleInPlace(g, s)
        }
    }
    return s
}

// gated debug print
func debugf(format string, args ...any) {
    if !config.Debug {
        return
    }
    // timestamp to help correlate across goroutines
    ts := time.Now().Format("15:04:05.000")
    fmt.Printf("[DBG %s] %s\n", ts, fmt.Sprintf(format, args...))
}

// p -= lr * (mhat/(sqrt(vhat)+eps) + wd * p) with bias correction (AdamW).

// p -= lr * mhat / (sqrt(vhat)+eps) with bias correction.
func adamUpdateInPlace(
	p, g, m, v *mat.Dense,
	t int,
	lr, beta1, beta2, eps, weightDecay float64,
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
			denom := math.Sqrt(vhat) + eps
            wdTerm := weightDecay * p.At(i, j)
            update := mhat/denom + wdTerm
            pij := p.At(i, j) - lr*update
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