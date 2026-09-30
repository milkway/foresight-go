package foresight

import "math"

// Theta is the Theta method (Assimakopoulos & Nikolopoulos, 2000), in the
// form shown by Hyndman & Billah (2003) to be simple exponential smoothing
// with drift: the smoothed level plus half the slope of the linear trend.
//
// Seasonal series are first tested for seasonality (autocorrelation at the
// seasonal lag, 90% level); when the test passes and the values are positive,
// the series is seasonally adjusted by classical multiplicative decomposition
// and the forecasts are re-seasonalised. The smoothing parameter and the
// initial level minimise the sum of squared one-step errors.
type Theta struct{}

type thetaFit struct {
	level, alpha, initialLevel float64
	// half the slope of the linear trend
	drift float64
	n     int
	// seasonal indices by position, and the position of the first forecast
	seasonal []float64
	first    int
}

func (f thetaFit) Forecast(h int) []float64 {
	if h <= 0 {
		return nil
	}
	carry := (1 - math.Pow(1-f.alpha, float64(f.n))) / f.alpha
	out := make([]float64, h)
	for k := 1; k <= h; k++ {
		v := f.level + f.drift*(float64(k-1)+carry)
		if f.seasonal != nil {
			v *= f.seasonal[(f.first+k-1)%len(f.seasonal)]
		}
		out[k-1] = v
	}
	return out
}

func (f thetaFit) Params() []Param {
	seasonal := 0.0
	if f.seasonal != nil {
		seasonal = 1
	}
	return []Param{
		{"alpha", f.alpha},
		{"initial_level", f.initialLevel},
		{"drift", f.drift},
		{"seasonal", seasonal},
	}
}

// isSeasonal reports whether the autocorrelation at the seasonal lag is
// significant at 90% (Bartlett's standard error).
func isSeasonal(y []float64, m int) bool {
	r := ACF(y, m)
	others := 0.0
	for i, v := range r {
		if !finite(v) {
			return false
		}
		if i < m-1 {
			others += v * v
		}
	}
	limit := 1.645 * math.Sqrt((1+2*others)/float64(len(y)))
	return math.Abs(r[m-1]) > limit
}

// seasonalIndices returns the seasonal indices (by position in the cycle,
// position 0 being y[0]) of the classical multiplicative decomposition:
// ratios to a centred moving average, averaged by position and scaled to
// mean 1.
func seasonalIndices(y []float64, m int) ([]float64, bool) {
	n := len(y)
	half := m / 2
	total := make([]float64, m)
	count := make([]int, m)
	for t := half; t < n-half; t++ {
		total[t%m] += y[t] / centredAverage(y, t, m)
		count[t%m]++
	}
	raw := make([]float64, m)
	for i := range raw {
		if count[i] == 0 {
			return nil, false
		}
		raw[i] = total[i] / float64(count[i])
	}
	mean := sum(raw) / float64(m)
	for i := range raw {
		raw[i] /= mean
		if !finite(raw[i]) || raw[i] <= 1e-4 {
			return nil, false
		}
	}
	return raw, true
}

// ses runs simple exponential smoothing for a given α with the initial level
// that minimises the sum of squared one-step errors (the errors are linear
// in the initial level, so it has a closed form).
func ses(y []float64, alpha float64) (sse, initial, level float64) {
	// level before y[t] = u + w·l0; error = (y[t] − u) − w·l0
	u, w := 0.0, 1.0
	var sxy, sxx float64
	for _, v := range y {
		sxy += w * (v - u)
		sxx += w * w
		u = alpha*v + (1-alpha)*u
		w *= 1 - alpha
	}
	initial = sxy / sxx
	level = initial
	for _, v := range y {
		e := v - level
		sse += e * e
		level += alpha * e
	}
	return sse, initial, level
}

// bestAlpha searches α in [0.01, 0.99]: a grid of 99 points refined by
// golden section around the best one.
func bestAlpha(y []float64) float64 {
	sse := func(a float64) float64 { s, _, _ := ses(y, a); return s }
	bestValue, best := math.Inf(1), 0.5
	for i := 1; i <= 99; i++ {
		a := float64(i) / 100
		if s := sse(a); s < bestValue {
			bestValue, best = s, a
		}
	}
	lo, hi := math.Max(best-0.01, 0.0001), math.Min(best+0.01, 0.9999)
	ratio := (math.Sqrt(5) - 1) / 2
	for range 40 {
		a, b := hi-ratio*(hi-lo), lo+ratio*(hi-lo)
		if sse(a) < sse(b) {
			hi = b
		} else {
			lo = a
		}
	}
	return (lo + hi) / 2
}

// Name is the identifier of the model.
func (Theta) Name() string { return "theta" }

// Description is a one-line description of the model.
func (Theta) Description() string {
	return "Theta method: exponential smoothing with drift, on seasonally adjusted data when seasonal"
}

// Fit estimates the model; it needs three observations.
func (Theta) Fit(y Series) (Fitted, error) {
	v, m, n := y.Values(), y.Period(), y.Len()
	if n < 3 {
		return nil, ErrTooShort
	}
	if !y.IsFinite() {
		return nil, ErrNotFinite
	}
	constant := true
	for _, x := range v {
		if x != v[0] {
			constant = false
			break
		}
	}
	var indices []float64
	if m > 1 && n > 2*m && !constant && y.IsPositive() && isSeasonal(v, m) {
		indices, _ = seasonalIndices(v, m)
	}
	adjusted := make([]float64, n)
	for t, x := range v {
		if indices != nil {
			adjusted[t] = x / indices[t%m]
		} else {
			adjusted[t] = x
		}
	}
	alpha := bestAlpha(adjusted)
	_, initial, level := ses(adjusted, alpha)
	_, slope, ok := line(adjusted)
	if !ok || !finite(level) || !finite(initial) {
		return nil, ErrNoFit
	}
	return thetaFit{
		level: level, alpha: alpha, initialLevel: initial,
		drift: slope / 2, n: n, seasonal: indices, first: n % m,
	}, nil
}
