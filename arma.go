package foresight

import "math"

// arma is X(t) = φ₁X(t−1) + … + φₚX(t−p) + Z(t) + θ₁Z(t−1) + … + θ_qZ(t−q).
// The exact Gaussian likelihood and the forecasts come from the innovations
// algorithm (Brockwell & Davis, Time Series: Theory and Methods, §5.2–5.3
// and §8.7).
type arma struct {
	phi, theta []float64
}

// innovations are the coefficients θ(n, j) and the mean squared errors v(n)
// of the one-step predictors, in units of the innovation variance.
type innovations struct {
	// theta[n][j] = θ(n, j), j = 1..min(n, width)
	theta [][]float64
	v     []float64
}

// filtered is what the likelihood needs from one pass over the data.
type filtered struct {
	// one-step prediction errors x(t) − x̂(t)
	errors []float64
	// Σ errors² / v
	sumSquares float64
	// Σ ln v
	logDet float64
}

func (a arma) m() int { return max(len(a.phi), len(a.theta)) }

// thetaAt is θ with θ₀ = 1 and zeros past q.
func (a arma) thetaAt(j int) float64 {
	switch {
	case j == 0:
		return 1
	case j <= len(a.theta):
		return a.theta[j-1]
	}
	return 0
}

// psi returns the coefficients ψ₀..ψₙ of the MA(∞) representation.
func (a arma) psi(n int) []float64 {
	psi := make([]float64, n+1)
	psi[0] = 1
	for j := 1; j <= n; j++ {
		ar := 0.0
		for k := 1; k <= min(j, len(a.phi)); k++ {
			ar += a.phi[k-1] * psi[j-k]
		}
		psi[j] = a.thetaAt(j) + ar
	}
	return psi
}

func absDiff(a, b int) int {
	if a > b {
		return a - b
	}
	return b - a
}

// acvf returns the autocovariances γ(0)..γ(maxLag) for unit innovation
// variance.
func (a arma) acvf(maxLag int) ([]float64, bool) {
	p, q := len(a.phi), len(a.theta)
	psi := a.psi(q)
	// γ(k) − Σ φⱼ γ(|k − j|) = Σ_{j ≥ k} θⱼ ψ(j − k)
	rhs := func(k int) float64 {
		s := 0.0
		for j := k; j <= q; j++ {
			s += a.thetaAt(j) * psi[j-k]
		}
		return s
	}
	mat := make([][]float64, p+1)
	b := make([]float64, p+1)
	for k := 0; k <= p; k++ {
		mat[k] = make([]float64, p+1)
	}
	for k := 0; k <= p; k++ {
		mat[k][k]++
		for j := 1; j <= p; j++ {
			mat[k][absDiff(k, j)] -= a.phi[j-1]
		}
		b[k] = rhs(k)
	}
	gamma, ok := solve(mat, b)
	if !ok {
		return nil, false
	}
	for k := p + 1; k <= maxLag; k++ {
		ar := 0.0
		for j := 1; j <= p; j++ {
			ar += a.phi[j-1] * gamma[k-j]
		}
		gamma = append(gamma, ar+rhs(k))
	}
	return gamma[:maxLag+1], true
}

// innovate returns the innovations coefficients for predictors 1..last.
func (a arma) innovate(last int) (innovations, bool) {
	p, q, m := len(a.phi), len(a.theta), a.m()
	gamma, ok := a.acvf(2*m + p)
	if !ok {
		return innovations{}, false
	}
	ma := make([]float64, q+1)
	for k := 0; k <= q; k++ {
		for r := 0; r <= q-k; r++ {
			ma[k] += a.thetaAt(r) * a.thetaAt(r+k)
		}
	}
	// covariance of the transformed process, indices from 1
	kappa := func(i, j int) float64 {
		lo, hi := min(i, j), max(i, j)
		k := hi - lo
		switch {
		case hi <= m:
			return gamma[k]
		case lo <= m:
			if hi > 2*m {
				return 0
			}
			ar := 0.0
			for r := 1; r <= p; r++ {
				ar += a.phi[r-1] * gamma[absDiff(r, k)]
			}
			return gamma[k] - ar
		case k <= q:
			return ma[k]
		}
		return 0
	}
	theta := make([][]float64, 0, last+1)
	v := make([]float64, 0, last+1)
	theta = append(theta, []float64{0})
	v = append(v, kappa(1, 1))
	if math.IsNaN(v[0]) || v[0] <= 0 {
		return innovations{}, false
	}
	for n := 1; n <= last; n++ {
		first := max(n-m, 0)
		row := make([]float64, min(n, m)+1)
		for k := first; k < n; k++ {
			s := kappa(n+1, k+1)
			for j := first; j < k; j++ {
				s -= theta[k][k-j] * row[n-j] * v[j]
			}
			row[n-k] = s / v[k]
		}
		explained := 0.0
		for j := first; j < n; j++ {
			explained += row[n-j] * row[n-j] * v[j]
		}
		vn := kappa(n+1, n+1) - explained
		if !finite(vn) || vn <= 0 {
			return innovations{}, false
		}
		// past max(p, q) the covariances depend only on the lag, so the
		// coefficients settle; once they have, the rest is a copy
		settled := n > 2*m && math.Abs(vn-v[n-1]) < 1e-14
		if settled {
			for i := range min(len(row), len(theta[n-1])) {
				if math.Abs(row[i]-theta[n-1][i]) >= 1e-14 {
					settled = false
					break
				}
			}
		}
		theta = append(theta, row)
		v = append(v, vn)
		if settled {
			for range last - n {
				theta = append(theta, row)
				v = append(v, vn)
			}
			break
		}
	}
	return innovations{theta, v}, true
}

// filter returns the one-step predictions of x and the pieces of the
// likelihood.
func (a arma) filter(x []float64, inn innovations) filtered {
	p, q, m := len(a.phi), len(a.theta), a.m()
	f := filtered{errors: make([]float64, 0, len(x))}
	for n := range x {
		// predictor of x[n] from x[:n]
		predicted := 0.0
		if n < m {
			for j := 1; j <= n; j++ {
				predicted += inn.theta[n][j] * f.errors[n-j]
			}
		} else {
			ar, ma := 0.0, 0.0
			for i := 1; i <= p; i++ {
				ar += a.phi[i-1] * x[n-i]
			}
			for j := 1; j <= q; j++ {
				ma += inn.theta[n][j] * f.errors[n-j]
			}
			predicted = ar + ma
		}
		e := x[n] - predicted
		f.sumSquares += e * e / inn.v[n]
		f.logDet += math.Log(inn.v[n])
		f.errors = append(f.errors, e)
	}
	return f
}

// forecast returns the forecasts of the h values after x, given its one-step
// errors. It needs len(x) ≥ max(p, q) and innovations up to len(x) + h − 1.
func (a arma) forecast(x, errors []float64, inn innovations, h int) []float64 {
	p, q, n := len(a.phi), len(a.theta), len(x)
	out := make([]float64, 0, h)
	for k := 1; k <= h; k++ {
		ar, ma := 0.0, 0.0
		for i := 1; i <= p; i++ {
			value := 0.0
			if i < k {
				value = out[k-1-i]
			} else {
				value = x[n+k-1-i]
			}
			ar += a.phi[i-1] * value
		}
		for j := k; j <= q; j++ {
			ma += inn.theta[n+k-1][j] * errors[n+k-1-j]
		}
		out = append(out, ar+ma)
	}
	return out
}

// stationary returns the coefficients of a stationary autoregression from
// unconstrained numbers: each goes through tanh to a partial autocorrelation
// in (−1, 1), and the Durbin-Levinson recursion turns those into coefficients
// (Jones, 1980).
func stationary(u []float64) []float64 {
	phi := make([]float64, 0, len(u))
	for k, x := range u {
		r := math.Tanh(x)
		previous := append([]float64(nil), phi...)
		for j := 0; j < k; j++ {
			phi[j] = previous[j] - r*previous[k-1-j]
		}
		phi = append(phi, r)
	}
	return phi
}

// rootsOutside reports whether 1 − φ₁z − … − φₚzᵖ has every root farther from
// the origin than margin (1 for plain stationarity), by the step-down
// recursion on the coefficients scaled by marginʲ.
func rootsOutside(phi []float64, margin float64) bool {
	c := make([]float64, len(phi))
	for j, v := range phi {
		c[j] = v * math.Pow(margin, float64(j+1))
	}
	for len(c) > 0 {
		r := c[len(c)-1]
		c = c[:len(c)-1]
		if math.IsNaN(r) || math.Abs(r) >= 1 {
			return false
		}
		k := len(c)
		previous := append([]float64(nil), c...)
		for j := 0; j < k; j++ {
			c[j] = (previous[j] + r*previous[k-1-j]) / (1 - r*r)
		}
	}
	return true
}

// expand multiplies the polynomials 1 + a₁z + … and 1 + b₁zˢ + b₂z²ˢ + …; it
// returns the coefficients of z, z², … of the result.
func expand(a, b []float64, s int) []float64 {
	out := make([]float64, len(a)+len(b)*s+1)
	out[0] = 1
	copy(out[1:], a)
	base := append([]float64(nil), out...)
	for j, w := range b {
		shift := (j + 1) * s
		for i := 0; i <= len(a); i++ {
			out[i+shift] += base[i] * w
		}
	}
	return out[1:]
}

func negate(v []float64) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = -x
	}
	return out
}
