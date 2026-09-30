package foresight

import "math"

var (
	hwAlpha = []float64{0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95}
	hwBeta  = []float64{0.0, 0.02, 0.05, 0.1, 0.2}
	hwGamma = []float64{0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8}
	hwPhi   = []float64{0.9, 0.95, 0.98, 1.0}
)

// HoltWinters is exponential smoothing with level, damped additive trend (φ)
// and multiplicative seasonal indices.
//
// α, β, γ and φ come from a grid search (3,040 combinations) minimising the
// sum of squared relative one-step-ahead errors on the training data, the
// first cycle left out as warm-up. It needs a seasonal period of at least 2,
// positive values and three full cycles.
type HoltWinters struct{}

type hwState struct {
	level, trend float64
	seasonal     []float64
}

// Classical starting values: level = mean of the first cycle; trend =
// difference between the means of the second and first cycles ÷ period;
// seasonal index = average, over the first two cycles, of y ÷ cycle mean.
func hwInitial(y []float64, m int) hwState {
	m1 := sum(y[:m]) / float64(m)
	m2 := sum(y[m:2*m]) / float64(m)
	s := make([]float64, m)
	for j := range s {
		s[j] = (y[j]/m1 + y[m+j]/m2) / 2
	}
	return hwState{level: m1, trend: (m2 - m1) / float64(m), seasonal: s}
}

// hwFilter runs the recursions over y from the state e, leaving the final
// state in it; it returns the sum of squared relative one-step errors from
// the second cycle on.
func hwFilter(y []float64, m int, e *hwState, a, b, g, phi float64) (float64, bool) {
	sse := 0.0
	for t, obs := range y {
		s := t % m
		base := e.level + phi*e.trend
		pred := base * e.seasonal[s]
		if t >= m {
			r := (obs - pred) / obs
			sse += r * r
		}
		level := a*obs/e.seasonal[s] + (1-a)*base
		if !finite(level) || level <= 0 {
			return 0, false
		}
		e.trend = b*(level-e.level) + (1-b)*phi*e.trend
		e.seasonal[s] = g*obs/level + (1-g)*e.seasonal[s]
		e.level = level
	}
	return sse, finite(sse)
}

type hwFit struct {
	alpha, beta, gamma, phi, sse float64
	state                        hwState
	n                            int
}

// ŷ(T+k) = (L + (φ + φ² + … + φᵏ)·B) × S(season of T+k).
func (f hwFit) Forecast(h int) []float64 {
	m := len(f.state.seasonal)
	damp, pow := 0.0, 1.0
	out := make([]float64, h)
	for k := 1; k <= h; k++ {
		pow *= f.phi
		damp += pow
		t := f.n - 1 + k
		out[k-1] = math.Max((f.state.level+damp*f.state.trend)*f.state.seasonal[t%m], 0)
	}
	return out
}

func (f hwFit) Params() []Param {
	return []Param{
		{"alpha", f.alpha}, {"beta", f.beta}, {"gamma", f.gamma},
		{"phi", f.phi}, {"sse_relative", f.sse},
	}
}

func (HoltWinters) Name() string { return "holt_winters" }
func (HoltWinters) Description() string {
	return "Holt-Winters, multiplicative seasonality and damped trend (grid search on one-step error)"
}

func (HoltWinters) Fit(y Series) (Fitted, error) {
	v, m := y.Values(), y.Period()
	if m < 2 {
		return nil, ErrNotSeasonal
	}
	if len(v) < 3*m {
		return nil, ErrTooShort
	}
	if !y.IsPositive() {
		return nil, ErrNotPositive
	}
	start := hwInitial(v, m)
	scratch := hwState{seasonal: make([]float64, m)}
	reset := func() {
		scratch.level, scratch.trend = start.level, start.trend
		copy(scratch.seasonal, start.seasonal)
	}
	best := hwFit{sse: math.Inf(1)}
	found := false
	for _, a := range hwAlpha {
		for _, b := range hwBeta {
			for _, g := range hwGamma {
				for _, phi := range hwPhi {
					reset()
					if sse, ok := hwFilter(v, m, &scratch, a, b, g, phi); ok && (!found || sse < best.sse) {
						best.alpha, best.beta, best.gamma, best.phi, best.sse = a, b, g, phi, sse
						found = true
					}
				}
			}
		}
	}
	if !found {
		return nil, ErrNoFit
	}
	reset()
	if _, ok := hwFilter(v, m, &scratch, best.alpha, best.beta, best.gamma, best.phi); !ok {
		return nil, ErrNoFit
	}
	best.state, best.n = scratch, len(v)
	return best, nil
}
