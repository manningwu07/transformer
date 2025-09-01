package main

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// Matrix functions I'm going to use for the calculations in the program

// r = rows of matrix
// c = columns of matrix
// o = output
// m = matrix input number 1
// n = matrix input number 2

func dot(m, n mat.Matrix) mat.Matrix {
	r, _ := m.Dims()
	_, c := n.Dims()
	o := mat.NewDense(r, c, nil)
	o.Product(m, n)
	return o
}

func apply(fn func(i, j int, v float64) float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Apply(fn, m)
	return o
}

func scale(s float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Scale(s, m)
	return o
}

func multiply(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.MulElem(m, n)
	return o
}

func add(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Add(m, n)
	return o
}

func subtract(m, n mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	o := mat.NewDense(r, c, nil)
	o.Sub(m, n)
	return o
}

func addScalar(i float64, m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	a := make([]float64, r*c)
	for x := 0; x < r*c; x++ {
		a[x] = i
	}
	n := mat.NewDense(r, c, a)
	return add(m, n)
}

// -------- GELU activation (GPT-style) --------
// Approximation used by GPT-2/3:
// gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
// We provide:
// - geluApply: shape-compatible with mat.Dense.Apply (i,j,v) -> value
// - geluPrime: elementwise derivative given pre-activation matrix X

func geluApply(i, j int, x float64) float64 {
	const k = 0.7978845608028654 // sqrt(2/pi)
	t := k * (x + 0.044715*x*x*x)
	return 0.5 * x * (1.0 + math.Tanh(t))
}

func geluPrime(m mat.Matrix) *mat.Dense {
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

func addBias(m, bias *mat.Dense) *mat.Dense {
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

func lastCol(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	out := mat.NewDense(r, 1, nil)
	for i := 0; i < r; i++ {
		out.Set(i, 0, m.At(i, c-1))
	}
	return out
}

// causalMask returns (T x T) with 0 on and below diagonal, -Inf above.
func causalMask(T int) *mat.Dense {
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
func softmaxBackward(dA mat.Matrix, A *mat.Dense) *mat.Dense {
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

func CrossEntropyWithGrad(logits, target *mat.Dense) (float64, *mat.Dense) {
	prob := ColVectorSoftmax(logits)
	loss := 0.0
	r, _ := prob.Dims()
	grad := mat.NewDense(r, 1, nil)
	for i := 0; i < r; i++ {
		p := prob.At(i, 0)
		t := target.At(i, 0)
		if t == 1.0 {
			loss -= math.Log(p + 1e-12)
		}
		grad.Set(i, 0, p-t)
	}
	return loss, grad
}
