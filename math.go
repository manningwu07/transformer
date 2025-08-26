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

func sigmoid(r, c int, z float64) float64 {
	return 1.0 / (1 + math.Exp(-1*z))
}

func sigmoidPrime(m mat.Matrix) mat.Matrix {
	r, c := m.Dims()
	out := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
	for j := 0; j < c; j++ {
			v := m.At(i, j)
			out.Set(i, j, v*(1.0-v))
		}
	}
	return out
}

// ---------- Softmax variants ----------

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
func softmaxBackward(dA, A *mat.Dense) *mat.Dense {
	r, c := A.Dims()
	dS := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		// Jacobian-vector product per row
		for j := 0; j < c; j++ {
			grad := 0.0
			for k := 0; k < c; k++ {
				if j == k {
					grad += A.At(i, j) * (1.0 - A.At(i, k)) * dA.At(i, k)
				} else {
					grad += -A.At(i, j) * A.At(i, k) * dA.At(i, k)
				}
			}
			dS.Set(i, j, grad)
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
