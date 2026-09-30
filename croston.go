package foresight

// Intermittent says how the demand rate of an intermittent series is
// estimated.
type Intermittent int

const (
	// CrostonMethod (Croston, 1972): the size of the demands and the
	// interval between them are smoothed separately; the rate is
	// size ÷ interval.
	CrostonMethod Intermittent = iota
	// SBA (Syntetos & Boylan, 2005): Croston's rate times 1 − α/2, which
	// removes its upward bias.
	SBA
	// TSB (Teunter, Syntetos & Babai, 2011): the probability of a demand is
	// smoothed at every period, so the rate falls while nothing is sold.
	TSB
)

// Croston forecasts the demand per period of intermittent series, where most
// periods have no demand at all.
//
// The values must not be negative. The forecast is the same for every
// horizon: the rate at which demand is expected to arrive.
type Croston struct {
	Variant Intermittent
	// Alpha is the smoothing of the demand sizes (and of the intervals),
	// above 0 and up to 1 (default 0.1).
	Alpha float64
	// Beta is the smoothing of the probability of demand, for TSB (default:
	// the same as Alpha).
	Beta float64
	// Optimised chooses the smoothing that minimises the squared error of
	// the rate against what was demanded, instead of a fixed value.
	Optimised bool
}

type crostonRun struct {
	size, interval, probability, mse float64
}

func (c Croston) rate(alpha, size, interval, probability float64) float64 {
	switch c.Variant {
	case SBA:
		return (1 - alpha/2) * size / interval
	case TSB:
		return probability * size
	}
	return size / interval
}

// run returns the state after the last observation, for given smoothing
// parameters, and the mean squared error of the rate as a one-step forecast.
// It reports false when there is no demand at all.
func (c Croston) run(y []float64, alpha, beta float64) (crostonRun, bool) {
	first := -1
	for i, v := range y {
		if v > 0 {
			first = i
			break
		}
	}
	if first < 0 {
		return crostonRun{}, false
	}
	r := crostonRun{size: y[first], interval: float64(first + 1), probability: 1 / float64(first+1)}
	since, squares, n := 0.0, 0.0, 0
	for _, v := range y[first+1:] {
		e := v - c.rate(alpha, r.size, r.interval, r.probability)
		squares += e * e
		n++
		since++
		if v > 0 {
			r.size += alpha * (v - r.size)
			r.interval += alpha * (since - r.interval)
			r.probability += beta * (1 - r.probability)
			since = 0
		} else {
			r.probability *= 1 - beta
		}
	}
	if n > 0 {
		r.mse = squares / float64(n)
	}
	return r, true
}

type crostonFit struct {
	rate, alpha, beta float64
	state             crostonRun
	variant           Intermittent
}

func (f crostonFit) Forecast(h int) []float64 { return repeat(f.rate, h) }

func (f crostonFit) Params() []Param {
	p := []Param{{"alpha", f.alpha}, {"demand_size", f.state.size}}
	if f.variant == TSB {
		return append(p, Param{"beta", f.beta}, Param{"demand_probability", f.state.probability})
	}
	return append(p, Param{"demand_interval", f.state.interval})
}

func (c Croston) Name() string {
	return [...]string{"croston", "croston_sba", "croston_tsb"}[c.Variant]
}

func (c Croston) Description() string {
	return [...]string{
		"Croston: demand size ÷ interval between demands, each smoothed",
		"Croston with the Syntetos-Boylan correction of bias",
		"Teunter-Syntetos-Babai: demand size × probability of demand, each smoothed",
	}[c.Variant]
}

func (c Croston) Fit(y Series) (Fitted, error) {
	v := y.Values()
	if len(v) == 0 {
		return nil, ErrTooShort
	}
	for _, x := range v {
		if !finite(x) {
			return nil, ErrNotFinite
		}
		if x < 0 {
			return nil, ErrNotPositive
		}
	}
	alpha := c.Alpha
	if alpha == 0 {
		alpha = 0.1
	}
	if c.Optimised {
		alpha = golden(func(a float64) float64 {
			beta := c.Beta
			if beta == 0 {
				beta = a
			}
			if r, ok := c.run(v, a, beta); ok {
				return r.mse
			}
			return inf
		}, 0.01, 0.99)
	}
	beta := c.Beta
	if beta == 0 {
		beta = alpha
	}
	inside := func(a float64) bool { return a > 0 && a <= 1 }
	if !inside(alpha) || !inside(beta) {
		return nil, ErrNoFit
	}
	state, ok := c.run(v, alpha, beta)
	if !ok {
		// no demand at all: nothing is expected
		return crostonFit{alpha: alpha, beta: beta, state: crostonRun{interval: inf}, variant: c.Variant}, nil
	}
	return crostonFit{
		rate:  c.rate(alpha, state.size, state.interval, state.probability),
		alpha: alpha, beta: beta, state: state, variant: c.Variant,
	}, nil
}
