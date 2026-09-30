package foresight

import "math"

// LogLinear is a log-linear regression: log(y/I) = a + b·t + seasonal
// dummies, by least squares over the last Window observations, where I is a
// price index (1 without a deflator).
//
// The back-transformation is corrected for bias with Duan's smearing factor
// (the mean of exp(residual)). With a deflator, real forecasts are
// re-inflated by the index growth over the last cycle, held constant over
// the horizon. It needs positive values and three full cycles.
type LogLinear struct {
	// Window is the number of most recent observations used in the fit.
	// Zero means six cycles, at least 24: enough to catch the recent trend
	// without dragging old breaks along.
	Window int
	// Deflator is a price index aligned with the ORIGINAL data: Deflator[i]
	// deflates the observation whose position is i (see [Series.Index]).
	// Only positions inside the training data are read, so handing the full
	// index to a backtest does not leak the future.
	Deflator []float64
}

type logLinearFit struct {
	// real (deflated) forecasts are exp(row(t)·beta) × duan
	beta        []float64
	duan        float64
	window      int
	period      int
	n           int
	firstSeason int
	center      float64
	deflated    bool
	base        float64
	inflation   float64
}

// columns: 1, t (in cycles, centred on the window), m−1 dummies
func logLinearRow(t, season int, center float64, m int) []float64 {
	x := make([]float64, m+1)
	x[0] = 1
	x[1] = (float64(t) - center) / float64(m)
	if season > 0 {
		x[1+season] = 1
	}
	return x
}

func (f logLinearFit) Forecast(h int) []float64 {
	if h <= 0 {
		return nil
	}
	m := f.period
	base, inflation := 1.0, 1.0
	if f.deflated {
		base, inflation = f.base, f.inflation
	}
	out := make([]float64, h)
	for k := 1; k <= h; k++ {
		t := f.n - 1 + k
		season := (f.firstSeason + k - 1) % m
		real := math.Exp(dot(logLinearRow(t, season, f.center, m), f.beta)) * f.duan
		out[k-1] = real * base * math.Pow(inflation, float64(k)/float64(m))
	}
	return out
}

func (f logLinearFit) Params() []Param {
	p := []Param{
		{"window", float64(f.window)},
		{"trend_growth_pct", (math.Exp(f.beta[1]) - 1) * 100},
		{"duan_factor", f.duan},
	}
	if f.deflated {
		p = append(p, Param{"inflation_pct", (f.inflation - 1) * 100})
	}
	return p
}

// Name is the identifier of the model.
func (l LogLinear) Name() string {
	if l.Deflator != nil {
		return "log_linear_deflated"
	}
	return "log_linear"
}

// Description is a one-line description of the model.
func (l LogLinear) Description() string {
	if l.Deflator != nil {
		return "Log-linear regression at constant prices, re-inflated by the index growth of the last cycle"
	}
	return "Log-linear regression: trend plus seasonal dummies"
}

// Fit estimates the model; it needs three full cycles of positive values
// and, with a deflator, an index that reaches the end of the series.
func (l LogLinear) Fit(y Series) (Fitted, error) {
	v, m, n := y.Values(), y.Period(), y.Len()
	if !y.IsPositive() {
		return nil, ErrNotPositive
	}
	window := l.Window
	if window <= 0 {
		window = max(6*m, 24)
	}
	w := min(n, window)
	if w < 3*m || w < m+2 {
		return nil, ErrTooShort
	}
	first := n - w
	var index []float64
	if l.Deflator != nil {
		from, to := y.Index(first), y.Index(n)
		if to > len(l.Deflator) {
			return nil, ErrLength
		}
		index = l.Deflator[from:to]
		for _, p := range index {
			if !finite(p) || p <= 0 {
				return nil, ErrNotPositive
			}
		}
	}
	price := func(t int) float64 {
		if index == nil {
			return 1
		}
		return index[t-first]
	}
	k := m + 1
	center := float64(first+n-1) / 2
	xtx := make([][]float64, k)
	for i := range xtx {
		xtx[i] = make([]float64, k)
	}
	xty := make([]float64, k)
	for t := first; t < n; t++ {
		x := logLinearRow(t, y.Season(t), center, m)
		target := math.Log(v[t] / price(t))
		for a := 0; a < k; a++ {
			xty[a] += x[a] * target
			for b := 0; b < k; b++ {
				xtx[a][b] += x[a] * x[b]
			}
		}
	}
	beta, ok := solve(xtx, xty)
	if !ok {
		return nil, ErrNoFit
	}
	duan := 0.0
	for t := first; t < n; t++ {
		duan += math.Exp(math.Log(v[t]/price(t)) - dot(logLinearRow(t, y.Season(t), center, m), beta))
	}
	duan /= float64(w)
	fit := logLinearFit{
		beta: beta, duan: duan, window: w, period: m, n: n,
		firstSeason: y.Season(n), center: center,
	}
	if index != nil {
		fit.deflated = true
		fit.base = index[w-1]
		fit.inflation = index[w-1] / index[w-1-m]
	}
	return fit, nil
}
