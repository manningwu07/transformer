package utils

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// Matrix functions I'm going to use for the calculations in the program

// r = rows of matrix
// c = columns of matrix
// o = output
// m = matrix input number 1
// n = matrix input number 2

func Dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func Apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

func Scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func Multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func Add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func Subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

func AddScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return Add(m, n)
}

func SampleFromProbs(probs *mat.Dense, topK int, topP float64) int {
	r, c := probs.Dims()
	if c != 1 {
		panic("sampleFromProbs expects column vector")
	}
	type kv struct {
		id  int
		val float64
	}
	arr := make([]kv, r)
	sum := 0.0
	for i := 0; i < r; i++ {
		p := probs.At(i, 0)
		arr[i] = kv{id: i, val: p}
		sum += p
	}
	// Normalize (just in case)
	for i := range arr {
		arr[i].val /= sum
	}

	// Sort descending by prob
	sort.Slice(arr, func(i, j int) bool { return arr[i].val > arr[j].val })

	// Apply top-k
	if topK > 0 && topK < len(arr) {
		arr = arr[:topK]
	}

	// Apply top-p (nucleus)
	if topP > 0 && topP < 1 {
		cum := 0.0
		cut := len(arr)
		for i, kv := range arr {
			cum += kv.val
			if cum >= topP {
				cut = i + 1
				break
			}
		}
		arr = arr[:cut]
	}

	// Renormalize after filtering
	sum = 0.0
	for _, kv := range arr {
		sum += kv.val
	}
	for i := range arr {
		arr[i].val /= sum
	}

	// Sample
	rnd := rand.Float64()
	cum := 0.0
	for _, kv := range arr {
		cum += kv.val
		if rnd < cum {
			return kv.id
		}
	}
	return arr[len(arr)-1].id // fallback
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

// -------- GELU activation (GPT-style) --------
// gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
// We provide:
// - geluApply: shape-compatible with mat.Dense.Apply (i,j,v) -> value
// - geluPrime: elementwise derivative given pre-activation matrix X

func GeluApply(i, j int, x float64) float64 {
	const k = 0.7978845608028654 // sqrt(2/pi)
	t := k * (x + 0.044715*x*x*x)
	return 0.5 * x * (1.0 + math.Tanh(t))
}

func GeluPrime(m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	out := mat.NewDense(r, c, nil)
	const k = 0.7978845608028654 // sqrt(2/pi)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			x := m.At(i, j)
			t := k * (x + 0.044715*x*x*x)
			th := math.Tanh(t)
			// sech^2(t) = 1 / cosh^2(t)
			cosh := math.Cosh(t)
			sech2 := 1.0 / (cosh * cosh)
			dt := k * (1.0 + 3.0*0.044715*x*x)
			// d/dx gelu approx:
			// 0.5*(1 + tanh(t)) + 0.5*x*sech^2(t)*dt
			grad := 0.5*(1.0+th) + 0.5*x*sech2*dt
			out.Set(i, j, grad)
		}
	}
	return out
}

// Masking stuff

func AddBias(m, bias *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	rb, cb := bias.Dims()
	if rb != r || cb != 1 {
		panic("addBias: bias must be (r x 1)")
	}
	out := mat.NewDense(r, c, nil)
	for j := 0; j < c; j++ {
		for i := 0; i < r; i++ {
			out.Set(i, j, m.At(i, j)+bias.At(i, 0))
		}
	}
	return out
}

func LastCol(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	out := mat.NewDense(r, 1, nil)
	for i := 0; i < r; i++ {
		out.Set(i, 0, m.At(i, c-1))
	}
	return out
}

// causalMask returns (T x T) with 0 on and below diagonal, -Inf above.
func CausalMask(T int) *mat.Dense {
	out := mat.NewDense(T, T, nil)
	negInf := -1e30
	for i := 0; i < T; i++ {
		for j := 0; j < T; j++ {
			if j > i {
				out.Set(i, j, negInf)
			} else {
				out.Set(i, j, 0.0)
			}
		}
	}
	return out
}

// ---------- Softmax variants ----------

// RowSoftmaxMaskedInPlace writes softmax(m+mask) into dst (r x c) in place
func RowSoftmaxMaskedInPlace(dst, m, mask *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	if dr, dc := dst.Dims(); dr != r || dc != c {
		panic("RowSoftmaxMaskedInPlace: dst shape mismatch")
	}
	if mr, mc := mask.Dims(); mr != r || mc != c {
		panic("RowSoftmaxMaskedInPlace: mask shape mismatch")
	}
	for i := 0; i < r; i++ {
		mx := m.At(i, 0) + mask.At(i, 0)
		for j := 1; j < c; j++ {
			v := m.At(i, j) + mask.At(i, j)
			if v > mx {
				mx = v
			}
		}
		sum := 0.0
		for j := 0; j < c; j++ {
			e := math.Exp(m.At(i, j) + mask.At(i, j) - mx)
			dst.Set(i, j, e)
			sum += e
		}
		inv := 1.0 / sum
		for j := 0; j < c; j++ {
			dst.Set(i, j, dst.At(i, j)*inv)
		}
	}
	return dst
}

// RowSoftmax applies softmax independently to each row across columns.
// Used by attention (scores have shape [d x d]; row sums should be 1).
func RowSoftmax(m mat.Matrix) *mat.Dense {
	r, c := m.Dims()
	out := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		// collect row
		row := make([]float64, c)
		for j := 0; j < c; j++ {
			row[j] = m.At(i, j)
		}
		// numerical stability
		mx := row[0]
		for _, v := range row {
			if v > mx {
				mx = v
			}
		}
		sum := 0.0
		for j := 0; j < c; j++ {
			row[j] = math.Exp(row[j] - mx)
			sum += row[j]
		}
		for j := 0; j < c; j++ {
			out.Set(i, j, row[j]/sum)
		}
	}
	return out
}

// ColVectorSoftmax applies softmax across the single column of a (r x 1) vector.
// Used for logits -> probabilities in the CE loss.
func ColVectorSoftmax(v *mat.Dense) *mat.Dense {
	r, c := v.Dims()
	if c != 1 {
		panic("ColVectorSoftmax expects a (r x 1) column vector")
	}
	out := mat.NewDense(r, 1, nil)
	// stability: subtract max
	mx := v.At(0, 0)
	for i := 1; i < r; i++ {
		if v.At(i, 0) > mx {
			mx = v.At(i, 0)
		}
	}
	sum := 0.0
	for i := 0; i < r; i++ {
		e := math.Exp(v.At(i, 0) - mx)
		out.Set(i, 0, e)
		sum += e
	}
	for i := 0; i < r; i++ {
		out.Set(i, 0, out.At(i, 0)/sum)
	}
	return out
}

// Softmax backward for row-wise softmax used in attention.
// Vector-JVP form: for each row i,
// s = sum_k dA[i,k] * A[i,k]; dS[i,j] = A[i,j] * (dA[i,j] - s)
func SoftmaxBackward(dA mat.Matrix, A *mat.Dense) *mat.Dense {
	r, c := A.Dims()
	dS := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		s := 0.0
		for k := 0; k < c; k++ {
			s += dA.At(i, k) * A.At(i, k)
		}
		for j := 0; j < c; j++ {
			aj := A.At(i, j)
			dS.Set(i, j, aj*(dA.At(i, j)-s))
		}
	}
	return dS
}

// ---------- Loss ----------

func CrossEntropyWithIndex(logits *mat.Dense, gold int) (float64, *mat.Dense) {
	r, c := logits.Dims()
	if c != 1 {
		panic("CrossEntropyWithIndex expects (r x 1) logits vector")
	}
	prob := ColVectorSoftmax(logits)
	if gold < 0 || gold >= r {
		gold = 0
	}
	loss := -math.Log(prob.At(gold, 0) + 1e-12)
	grad := mat.NewDense(r, 1, nil)
	for i := 0; i < r; i++ {
		grad.Set(i, 0, prob.At(i, 0))
	}
	grad.Set(gold, 0, grad.At(gold, 0)-1.0)
	return loss, grad
}
