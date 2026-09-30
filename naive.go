package foresight

import "math"

type constant float64

func (c constant) Forecast(h int) []float64 {
	out := make([]float64, h)
	for i := range out {
		out[i] = float64(c)
	}
	return out
}

func (constant) Params() []Param { return nil }

// Mean forecasts the mean of the history at every horizon.
type Mean struct{}

func (Mean) Name() string        { return "mean" }
func (Mean) Description() string { return "Mean of the history" }

func (Mean) Fit(y Series) (Fitted, error) {
	if y.Len() == 0 {
		return nil, ErrTooShort
	}
	if !y.IsFinite() {
		return nil, ErrNotFinite
	}
	return constant(sum(y.Values()) / float64(y.Len())), nil
}

// Naive forecasts the last observation at every horizon (a random walk).
type Naive struct{}

func (Naive) Name() string        { return "naive" }
func (Naive) Description() string { return "Naive: the last observation" }

func (Naive) Fit(y Series) (Fitted, error) {
	if y.Len() == 0 {
		return nil, ErrTooShort
	}
	if !y.IsFinite() {
		return nil, ErrNotFinite
	}
	return constant(y.Values()[y.Len()-1]), nil
}

// Drift is the random walk with drift: the last observation plus the average
// change per period over the history.
type Drift struct{}

type driftFit struct{ last, slope float64 }

func (f driftFit) Forecast(h int) []float64 {
	out := make([]float64, h)
	for k := range out {
		out[k] = f.last + float64(k+1)*f.slope
	}
	return out
}

func (f driftFit) Params() []Param { return []Param{{"drift", f.slope}} }

func (Drift) Name() string { return "drift" }
func (Drift) Description() string {
	return "Random walk with drift: last observation plus the average change"
}

func (Drift) Fit(y Series) (Fitted, error) {
	v := y.Values()
	if len(v) < 2 {
		return nil, ErrTooShort
	}
	if !y.IsFinite() {
		return nil, ErrNotFinite
	}
	last := v[len(v)-1]
	return driftFit{last: last, slope: (last - v[0]) / float64(len(v)-1)}, nil
}

// SeasonalNaive forecasts the same season of the last cycle, optionally
// scaled by recent growth.
//
// With Growth, ŷ(T+k) = y(T+k−m) × g, where g is the sum of the last cycle
// over the sum of the one before; forecasts beyond one cycle compound g.
type SeasonalNaive struct {
	// Growth scales the last cycle by the growth between the last two.
	Growth bool
}

type seasonalNaiveFit struct {
	lastCycle []float64
	growth    float64
	scaled    bool
}

func (f seasonalNaiveFit) Forecast(h int) []float64 {
	m := len(f.lastCycle)
	out := make([]float64, h)
	for k := 1; k <= h; k++ {
		cycles := (k-1)/m + 1
		g := 1.0
		if f.scaled {
			g = math.Pow(f.growth, float64(cycles))
		}
		out[k-1] = f.lastCycle[(k-1)%m] * g
	}
	return out
}

func (f seasonalNaiveFit) Params() []Param {
	if !f.scaled {
		return nil
	}
	return []Param{{"growth", f.growth}}
}

func (s SeasonalNaive) Name() string {
	if s.Growth {
		return "seasonal_naive_growth"
	}
	return "seasonal_naive"
}

func (s SeasonalNaive) Description() string {
	if s.Growth {
		return "Seasonal naive with growth: same season of the last cycle × growth between the last two cycles"
	}
	return "Seasonal naive: same season of the last cycle"
}

func (s SeasonalNaive) Fit(y Series) (Fitted, error) {
	v, m, n := y.Values(), y.Period(), y.Len()
	if !y.IsFinite() {
		return nil, ErrNotFinite
	}
	if n < m || (s.Growth && n < 2*m) {
		return nil, ErrTooShort
	}
	fit := seasonalNaiveFit{lastCycle: append([]float64(nil), v[n-m:]...)}
	if s.Growth {
		last := sum(v[n-m:])
		before := sum(v[n-2*m : n-m])
		if before <= 0 {
			return nil, ErrNotPositive
		}
		fit.growth, fit.scaled = last/before, true
	}
	return fit, nil
}
