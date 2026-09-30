package foresight

import "math"

// solve solves a·x = b by Gaussian elimination with partial pivoting. It
// reports false if the matrix is singular or the solution is not finite. The
// arguments are overwritten.
func solve(a [][]float64, b []float64) ([]float64, bool) {
	k := len(b)
	if len(a) != k {
		return nil, false
	}
	for col := 0; col < k; col++ {
		piv := col
		for i := col + 1; i < k; i++ {
			if math.Abs(a[i][col]) >= math.Abs(a[piv][col]) {
				piv = i
			}
		}
		if math.Abs(a[piv][col]) < 1e-12 {
			return nil, false
		}
		a[col], a[piv] = a[piv], a[col]
		b[col], b[piv] = b[piv], b[col]
		pivot := a[col]
		for row := col + 1; row < k; row++ {
			f := a[row][col] / pivot[col]
			if f != 0 {
				for c := col; c < k; c++ {
					a[row][c] -= f * pivot[c]
				}
				b[row] -= f * b[col]
			}
		}
	}
	x := make([]float64, k)
	for row := k - 1; row >= 0; row-- {
		s := 0.0
		for c := row + 1; c < k; c++ {
			s += a[row][c] * x[c]
		}
		x[row] = (b[row] - s) / a[row][row]
	}
	for _, v := range x {
		if !finite(v) {
			return nil, false
		}
	}
	return x, true
}

// line is the least squares fit of y = a + b·t, t = 0, 1, …
func line(y []float64) (a, b float64, ok bool) {
	n := len(y)
	if n < 2 {
		return 0, 0, false
	}
	nf := float64(n)
	tMean := (nf - 1) / 2
	yMean := sum(y) / nf
	var sxy, sxx float64
	for t, v := range y {
		d := float64(t) - tMean
		sxy += d * (v - yMean)
		sxx += d * d
	}
	b = sxy / sxx
	if !finite(b) {
		return 0, 0, false
	}
	return yMean - b*tMean, b, true
}

func sum(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s += x
	}
	return s
}

func dot(a, b []float64) float64 {
	s := 0.0
	for i := range min(len(a), len(b)) {
		s += a[i] * b[i]
	}
	return s
}
