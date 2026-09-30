package foresight

import (
	"math"
	"strings"
)

// ErrorKind says how the error enters the observation of an ETS model.
type ErrorKind int

const (
	// AdditiveError: y = μ + ε.
	AdditiveError ErrorKind = iota
	// MultiplicativeError: y = μ (1 + ε); needs positive values.
	MultiplicativeError
)

// Trend is the trend component of an ETS model.
type Trend int

const (
	NoTrend Trend = iota
	AdditiveTrend
	// DampedTrend is additive, flattening out over the horizon.
	DampedTrend
)

// Season is the seasonal component of an ETS model.
type Season int

const (
	NoSeason Season = iota
	AdditiveSeason
	// MultiplicativeSeason needs positive values.
	MultiplicativeSeason
)

// Ets is one member of the exponential smoothing family in state space form
// (Hyndman, Koehler, Snyder & Grose, 2002): an error, a trend and a seasonal
// component.
//
// Simple exponential smoothing is Ets{}, Holt's linear method
// Ets{Trend: AdditiveTrend}, the multiplicative Holt-Winters method
// Ets{MultiplicativeError, AdditiveTrend, MultiplicativeSeason}. The
// smoothing parameters and the initial states are estimated together by
// maximum likelihood.
type Ets struct {
	Error  ErrorKind
	Trend  Trend
	Season Season
}

// EtsFromCode returns the model of the usual three-letter code,
// error–trend–season, with Ad for a damped trend: "ANN", "AAdN", "MAM"… It
// reports false for anything else.
func EtsFromCode(code string) (Ets, bool) {
	var e Ets
	switch {
	case strings.HasPrefix(code, "A"):
	case strings.HasPrefix(code, "M"):
		e.Error = MultiplicativeError
	default:
		return Ets{}, false
	}
	rest := code[1:]
	switch {
	case strings.HasPrefix(rest, "Ad"):
		e.Trend, rest = DampedTrend, rest[2:]
	case strings.HasPrefix(rest, "A"):
		e.Trend, rest = AdditiveTrend, rest[1:]
	case strings.HasPrefix(rest, "N"):
		rest = rest[1:]
	default:
		return Ets{}, false
	}
	switch rest {
	case "N":
	case "A":
		e.Season = AdditiveSeason
	case "M":
		e.Season = MultiplicativeSeason
	default:
		return Ets{}, false
	}
	return e, true
}

// Code returns the three-letter code of the model.
func (e Ets) Code() string {
	code := "A"
	if e.Error == MultiplicativeError {
		code = "M"
	}
	code += [...]string{"N", "A", "Ad"}[e.Trend]
	return code + [...]string{"N", "A", "M"}[e.Season]
}

const (
	etsAlphaMin, etsAlphaMax = 1e-4, 0.9999
	etsBetaMin, etsGammaMin  = 1e-4, 1e-4
	etsPhiMin, etsPhiMax     = 0.8, 0.98
)

func logistic(u float64) float64 { return 1 / (1 + math.Exp(-u)) }
func logit(p float64) float64    { return math.Log(p / (1 - p)) }

// etsState has the parameters and the states of a model.
type etsState struct {
	alpha, beta, gamma, phi float64
	level, trend            float64
	// by position in the cycle; position 0 is the first observation
	seasonal []float64
}

func (s etsState) clone() etsState {
	s.seasonal = append([]float64(nil), s.seasonal...)
	return s
}

// etsPass is what one pass over the data gives.
type etsPass struct {
	// ε of each observation (relative for multiplicative errors)
	errors, fitted []float64
	// n ln Σε² + 2 Σ ln|μ| (the last term only for multiplicative errors)
	criterion float64
	// states after the last observation
	last etsState
}

func (e Ets) hasTrend() bool { return e.Trend != NoTrend }

func (e Ets) forPeriod(m int) Ets {
	if m < 2 {
		e.Season = NoSeason
	}
	return e
}

func count(b bool) int {
	if b {
		return 1
	}
	return 0
}

// parameters counts what is estimated: smoothing parameters, initial states
// and the variance.
func (e Ets) parameters(m int) int {
	smoothing := 1 + count(e.hasTrend()) + count(e.Trend == DampedTrend) + count(e.Season != NoSeason)
	states := 1 + count(e.hasTrend())
	if e.Season != NoSeason {
		states += m - 1
	}
	return smoothing + states + 1
}

// pass runs the recursions over y from the given states.
func (e Ets) pass(y []float64, start etsState) (etsPass, bool) {
	s := start.clone()
	m := max(len(s.seasonal), 1)
	p := etsPass{errors: make([]float64, 0, len(y)), fitted: make([]float64, 0, len(y))}
	squares, logs := 0.0, 0.0
	for t, obs := range y {
		at := t % m
		q := s.level + s.phi*s.trend
		season := 0.0
		if e.Season != NoSeason {
			season = s.seasonal[at]
		}
		mu := q
		switch e.Season {
		case AdditiveSeason:
			mu = q + season
		case MultiplicativeSeason:
			mu = q * season
		}
		if !finite(mu) {
			return etsPass{}, false
		}
		err := obs - mu
		if e.Error == MultiplicativeError {
			if mu <= 0 {
				return etsPass{}, false
			}
			logs += math.Log(mu)
			err = (obs - mu) / mu
		}
		// what a unit of error does to the level, the trend and the season
		toLevel, toSeason := 1.0, 1.0
		switch {
		case e.Error == AdditiveError && e.Season == MultiplicativeSeason:
			toLevel, toSeason = 1/season, 1/q
		case e.Error == MultiplicativeError && e.Season == NoSeason:
			toLevel, toSeason = q, 0
		case e.Error == MultiplicativeError && e.Season == AdditiveSeason:
			toLevel, toSeason = mu, mu
		case e.Error == MultiplicativeError && e.Season == MultiplicativeSeason:
			toLevel, toSeason = q, season
		}
		if !finite(toLevel) || !finite(toSeason) {
			return etsPass{}, false
		}
		s.level = q + s.alpha*toLevel*err
		s.trend = s.phi*s.trend + s.beta*toLevel*err
		if e.Season != NoSeason {
			s.seasonal[at] = season + s.gamma*toSeason*err
		}
		squares += err * err
		p.errors = append(p.errors, err)
		p.fitted = append(p.fitted, mu)
	}
	p.criterion = -1e300
	if squares > 0 {
		p.criterion = float64(len(y))*math.Log(squares) + 2*logs
	}
	p.last = s
	return p, finite(p.criterion)
}

// initial returns starting values of the states: seasonal pattern from a
// classical decomposition of the first cycles, level and trend from a line
// through the first seasonally adjusted observations.
func (e Ets) initial(y []float64, m int) (etsState, bool) {
	n := len(y)
	var seasonal []float64
	if e.Season != NoSeason {
		used := y[:min(n, 3*m)]
		half := m / 2
		total := make([]float64, m)
		counts := make([]int, m)
		for t := half; t < len(used)-half; t++ {
			centre := centredAverage(used, t, m)
			if e.Season == AdditiveSeason {
				total[t%m] += used[t] - centre
			} else {
				total[t%m] += used[t] / centre
			}
			counts[t%m]++
		}
		seasonal = make([]float64, m)
		for i := range seasonal {
			if counts[i] == 0 {
				return etsState{}, false
			}
			seasonal[i] = total[i] / float64(counts[i])
		}
		mean := sum(seasonal) / float64(m)
		for i := range seasonal {
			if e.Season == AdditiveSeason {
				seasonal[i] -= mean
			} else {
				seasonal[i] /= mean
			}
		}
	}
	adjusted := make([]float64, min(n, max(10, 2*m)))
	for t := range adjusted {
		switch e.Season {
		case AdditiveSeason:
			adjusted[t] = y[t] - seasonal[t%m]
		case MultiplicativeSeason:
			adjusted[t] = y[t] / seasonal[t%m]
		default:
			adjusted[t] = y[t]
		}
	}
	s := etsState{phi: 1, seasonal: seasonal}
	if e.hasTrend() {
		// the line gives the value at the first observation; the state is
		// one step before it
		a, b, ok := line(adjusted)
		if !ok {
			return etsState{}, false
		}
		s.level, s.trend = a-b, b
	} else {
		s.level = sum(adjusted) / float64(len(adjusted))
	}
	return s, true
}

// EtsFit is an estimated ETS model.
type EtsFit struct {
	spec   Ets
	period int
	n      int
	// Alpha is the smoothing of the level.
	Alpha float64
	// Beta is the smoothing of the trend, Gamma of the seasonal pattern and
	// Phi the damping of the trend; they are NaN when the model has no such
	// part.
	Beta, Gamma, Phi float64
	initial, last    etsState
	// Sigma2 is the variance of the errors (relative errors for
	// multiplicative models).
	Sigma2 float64
	// LogLikelihood is the log-likelihood up to the usual constant, as
	// reported by ets in R.
	LogLikelihood float64
	AIC           float64
	AICc          float64
	BIC           float64
	// Residuals are the errors of the one-step forecasts (relative for
	// multiplicative models).
	Residuals []float64
	// Fitted are the one-step forecasts of the history.
	Fitted []float64
}

// Estimate fits the model and returns everything that was estimated.
func (e Ets) Estimate(y Series) (*EtsFit, error) {
	m := y.Period()
	spec := e.forPeriod(m)
	multiplicative := spec.Error == MultiplicativeError || spec.Season == MultiplicativeSeason
	if !y.IsFinite() {
		return nil, ErrNotFinite
	}
	if multiplicative && !y.IsPositive() {
		return nil, ErrNotPositive
	}
	n := y.Len()
	k := spec.parameters(m)
	if n < k+4 || (spec.Season != NoSeason && n < 2*m) {
		return nil, ErrTooShort
	}
	// the models keep their shape under a change of units: work near 1
	scale := 0.0
	for _, v := range y.Values() {
		scale += math.Abs(v)
	}
	scale /= float64(n)
	if math.IsNaN(scale) || scale <= 0 {
		return nil, ErrNoFit
	}
	v := make([]float64, n)
	for i, x := range y.Values() {
		v[i] = x / scale
	}
	base, ok := spec.initial(v, m)
	if !ok {
		return nil, ErrTooShort
	}
	seasonal := spec.Season != NoSeason
	freeSeasons := 0
	if seasonal {
		freeSeasons = m - 1
	}

	// unconstrained numbers → parameters inside their bounds and states
	build := func(u []float64) etsState {
		i := 0
		take := func() float64 {
			if i < len(u) {
				i++
				return u[i-1]
			}
			return 0
		}
		s := etsState{phi: 1}
		s.alpha = etsAlphaMin + (etsAlphaMax-etsAlphaMin)*logistic(take())
		if spec.hasTrend() {
			s.beta = etsBetaMin + math.Max(s.alpha-etsBetaMin, 0)*logistic(take())
		}
		if seasonal {
			s.gamma = etsGammaMin + math.Max(1-s.alpha-etsGammaMin, 0)*logistic(take())
		}
		if spec.Trend == DampedTrend {
			s.phi = etsPhiMin + (etsPhiMax-etsPhiMin)*logistic(take())
		}
		s.level = base.level + 0.1*take()
		if spec.hasTrend() {
			s.trend = base.trend + 0.02*take()
		}
		if seasonal {
			s.seasonal = make([]float64, 0, m)
			total := 0.0
			for j := range freeSeasons {
				value := base.seasonal[j] + 0.05*take()
				s.seasonal = append(s.seasonal, value)
				total += value
			}
			// the pattern adds up to nothing (or averages one)
			if spec.Season == AdditiveSeason {
				s.seasonal = append(s.seasonal, -total)
			} else {
				s.seasonal = append(s.seasonal, float64(m)-total)
			}
		}
		return s
	}
	objective := func(u []float64) float64 {
		state := build(u)
		if spec.Season == MultiplicativeSeason {
			for _, s := range state.seasonal {
				if s <= 0 {
					return math.Inf(1)
				}
			}
		}
		p, ok := spec.pass(v, state)
		if !ok {
			return math.Inf(1)
		}
		return p.criterion
	}
	dimension := k - 1
	start := func(alpha, betaShare, gammaShare float64) []float64 {
		u := []float64{logit((alpha - etsAlphaMin) / (etsAlphaMax - etsAlphaMin))}
		if spec.hasTrend() {
			u = append(u, logit(betaShare))
		}
		if seasonal {
			u = append(u, logit(gammaShare))
		}
		if spec.Trend == DampedTrend {
			u = append(u, logit(0.9))
		}
		for len(u) < dimension {
			u = append(u, 0)
		}
		return u
	}
	var u []float64
	best := math.NaN()
	for i, s := range [][]float64{start(0.1, 0.1, 0.05), start(0.5, 0.1, 0.2), start(0.9, 0.05, 0.5)} {
		if x, value := nelderMead(objective, s, 1, thorough); i == 0 || below(value, best) {
			u, best = x, value
		}
	}
	state := build(u)
	pass, ok := spec.pass(v, state)
	if !ok {
		return nil, ErrNoFit
	}

	nf, kf := float64(n), float64(k)
	// back to the units of the data
	minusTwoLogLikelihood := pass.criterion + 2*nf*math.Log(scale)
	aic := minusTwoLogLikelihood + 2*kf
	aicc := math.Inf(1)
	if nf-kf-1 > 0 {
		aicc = aic + 2*kf*(kf+1)/(nf-kf-1)
	}
	unit := scale
	if spec.Error == MultiplicativeError {
		unit = 1
	}
	rescale := func(s etsState) etsState {
		s = s.clone()
		s.level *= scale
		s.trend *= scale
		if spec.Season == AdditiveSeason {
			for i := range s.seasonal {
				s.seasonal[i] *= scale
			}
		}
		return s
	}
	squares := 0.0
	for i := range pass.errors {
		squares += pass.errors[i] * pass.errors[i]
		pass.errors[i] *= unit
		pass.fitted[i] *= scale
	}
	fit := &EtsFit{
		spec: spec, period: m, n: n,
		Alpha: state.alpha, Beta: math.NaN(), Gamma: math.NaN(), Phi: math.NaN(),
		initial: rescale(state), last: rescale(pass.last),
		Sigma2:        squares * unit * unit / math.Max(nf-kf+1, 1),
		LogLikelihood: -0.5 * minusTwoLogLikelihood,
		AIC:           aic, AICc: aicc, BIC: minusTwoLogLikelihood + kf*math.Log(nf),
		Residuals: pass.errors, Fitted: pass.fitted,
	}
	if spec.hasTrend() {
		fit.Beta = state.beta
	}
	if seasonal {
		fit.Gamma = state.gamma
	}
	if spec.Trend == DampedTrend {
		fit.Phi = state.phi
	}
	return fit, nil
}

// Model returns the model that was fitted.
func (f *EtsFit) Model() Ets { return f.spec }

// InitialLevelAndTrend returns the level and the trend before the first
// observation.
func (f *EtsFit) InitialLevelAndTrend() (level, trend float64) {
	return f.initial.level, f.initial.trend
}

// InitialSeasonal returns the seasonal pattern before the first observation,
// by position in the cycle (position 0 is the first observation).
func (f *EtsFit) InitialSeasonal() []float64 { return f.initial.seasonal }

// LevelAndTrend returns the level and the trend after the last observation.
func (f *EtsFit) LevelAndTrend() (level, trend float64) {
	return f.last.level, f.last.trend
}

func (f *EtsFit) Forecast(h int) []float64 {
	m := max(f.period, 1)
	damp, pow := 0.0, 1.0
	out := make([]float64, h)
	for k := 1; k <= h; k++ {
		pow *= f.last.phi
		damp += pow
		q := f.last.level + damp*f.last.trend
		at := (f.n + k - 1) % m
		switch f.spec.Season {
		case AdditiveSeason:
			q += f.last.seasonal[at]
		case MultiplicativeSeason:
			q *= f.last.seasonal[at]
		}
		out[k-1] = q
	}
	return out
}

func (f *EtsFit) Params() []Param {
	p := []Param{{"alpha", f.Alpha}}
	if f.spec.hasTrend() {
		p = append(p, Param{"beta", f.Beta})
	}
	if f.spec.Season != NoSeason {
		p = append(p, Param{"gamma", f.Gamma})
	}
	if f.spec.Trend == DampedTrend {
		p = append(p, Param{"phi", f.Phi})
	}
	p = append(p, Param{"initial_level", f.initial.level})
	if f.spec.hasTrend() {
		p = append(p, Param{"initial_trend", f.initial.trend})
	}
	return append(p,
		Param{"multiplicative_error", float64(count(f.spec.Error == MultiplicativeError))},
		Param{"trend", float64(count(f.spec.hasTrend()))},
		Param{"damped", float64(count(f.spec.Trend == DampedTrend))},
		Param{"seasonal", float64(f.spec.Season)},
		Param{"sigma2", f.Sigma2},
		Param{"log_likelihood", f.LogLikelihood},
		Param{"aicc", f.AICc},
	)
}

func (e Ets) Name() string { return "ets_" + strings.ToLower(e.Code()) }

func (e Ets) Description() string {
	code := e.Code()
	return "Exponential smoothing ETS(" + code[:1] + "," + code[1:len(code)-1] + "," +
		code[len(code)-1:] + ") by maximum likelihood"
}

func (e Ets) Fit(y Series) (Fitted, error) {
	fit, err := e.Estimate(y)
	if err != nil {
		return nil, err
	}
	return fit, nil
}

// AutoEts is the ETS model with the best information criterion among those
// that suit the series.
//
// Every combination of error, trend (none, additive, damped) and season
// (none, additive, multiplicative) is fitted, leaving out the ones with
// multiplicative parts when the series has values that are not positive, and
// additive errors with a multiplicative season, whose forecast variance is
// unbounded.
//
// As a [Model], the choice is made again at every fit, so a backtest judges
// the whole procedure.
type AutoEts struct {
	Criterion Criterion
}

// EtsCandidates returns the models considered for a series of the given
// period.
func EtsCandidates(period int, positive bool) []Ets {
	var out []Ets
	for _, e := range []ErrorKind{AdditiveError, MultiplicativeError} {
		for _, t := range []Trend{NoTrend, AdditiveTrend, DampedTrend} {
			for _, s := range []Season{NoSeason, AdditiveSeason, MultiplicativeSeason} {
				multiplicative := e == MultiplicativeError || s == MultiplicativeSeason
				if (s != NoSeason && (period < 2 || period > 24)) ||
					(multiplicative && !positive) ||
					(e == AdditiveError && s == MultiplicativeSeason) {
					continue
				}
				out = append(out, Ets{e, t, s})
			}
		}
	}
	return out
}

// Select chooses the model and returns it estimated.
func (a AutoEts) Select(y Series) (*EtsFit, error) {
	score := func(f *EtsFit) float64 {
		switch a.Criterion {
		case ByAIC:
			return f.AIC
		case ByBIC:
			return f.BIC
		}
		return f.AICc
	}
	var best *EtsFit
	for _, m := range EtsCandidates(y.Period(), y.IsPositive()) {
		fit, err := m.Estimate(y)
		if err != nil || !finite(score(fit)) {
			continue
		}
		if best == nil || score(fit) < score(best) {
			best = fit
		}
	}
	if best == nil {
		return nil, ErrNoFit
	}
	return best, nil
}

func (AutoEts) Name() string { return "auto_ets" }
func (AutoEts) Description() string {
	return "Exponential smoothing with error, trend and season chosen by the information criterion"
}

func (a AutoEts) Fit(y Series) (Fitted, error) {
	fit, err := a.Select(y)
	if err != nil {
		return nil, err
	}
	return fit, nil
}
