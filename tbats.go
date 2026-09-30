package foresight

import (
	"math"
	"slices"
	"strconv"
)

// Choice is something a model decides by itself unless told.
type Choice int

const (
	// Choose lets the model decide.
	Choose Choice = iota
	// Yes and No fix the answer.
	Yes
	No
)

func (c Choice) or(fallback bool) bool {
	switch c {
	case Yes:
		return true
	case No:
		return false
	}
	return fallback
}

// allows reports whether the choice is open or agrees with the value.
func (c Choice) allows(value bool) bool { return c == Choose || c.or(false) == value }

// Tbats is TBATS (De Livera, Hyndman & Snyder, 2011): exponential smoothing
// with seasonal patterns made of sines and cosines.
//
// Because a pattern is a handful of harmonics rather than one state per
// position of the cycle, the period may be long (365 days) or not a whole
// number (52.18 weeks, 365.25 days), and several periods can be combined.
// What is left after level, trend and seasonality may follow an ARMA
// process, and the whole may run on a Box-Cox scale.
//
// Whatever is not fixed is chosen by AIC: the number of harmonics of each
// period, trend and damping, the transformation and the ARMA errors. The
// initial states are found by least squares for each set of parameters,
// which leaves the search with the few smoothing parameters only.
//
// It is the slowest model of the package: choosing the structure of a
// monthly series of ten years takes seconds.
type Tbats struct {
	// Periods are the seasonal periods, which need not be whole numbers.
	// With none, the period of the series is used.
	Periods []float64
	// Harmonics fixes the number of harmonics of each period, in increasing
	// order of period, instead of choosing them.
	Harmonics []int
	// BoxCox says whether the model runs on a Box-Cox scale (λ between 0 and
	// 1, estimated with the other parameters); Trend whether there is a
	// trend; Damped whether it is damped; ArmaErrors whether the errors may
	// follow an ARMA process.
	BoxCox, Trend, Damped, ArmaErrors Choice
	// FixOrders uses P and Q as the orders of the ARMA errors instead of
	// choosing them.
	FixOrders bool
	P, Q      int
}

// tbatsShape is what a model is made of.
type tbatsShape struct {
	periods               []float64
	harmonics             []int
	trend, damped, boxCox bool
	p, q                  int
}

func (s tbatsShape) clone() tbatsShape {
	s.periods = slices.Clone(s.periods)
	s.harmonics = slices.Clone(s.harmonics)
	return s
}

func (s tbatsShape) equal(o tbatsShape) bool {
	return slices.Equal(s.periods, o.periods) && slices.Equal(s.harmonics, o.harmonics) &&
		s.trend == o.trend && s.damped == o.damped && s.boxCox == o.boxCox && s.p == o.p && s.q == o.q
}

// tbatsValues are the parameters of a model.
type tbatsValues struct {
	alpha, beta, phi float64
	// (γ₁, γ₂) of each period
	gamma     [][2]float64
	ar, ma    []float64
	lambda    float64
	hasLambda bool
}

func (s tbatsShape) seasonalStates() int {
	total := 0
	for _, k := range s.harmonics {
		total += k
	}
	return 2 * total
}

// seeds are the states estimated from the data: level, trend and the
// harmonics.
func (s tbatsShape) seeds() int { return 1 + count(s.trend) + s.seasonalStates() }

func (s tbatsShape) states() int { return s.seeds() + s.p + s.q }

// dimension is how many numbers the search moves.
func (s tbatsShape) dimension() int {
	return 1 + count(s.trend) + count(s.damped) + 2*len(s.periods) + s.p + s.q + count(s.boxCox)
}

// counted are the parameters and states counted by the information
// criterion.
func (s tbatsShape) counted() int { return s.dimension() + s.states() }

func (s tbatsShape) start(lambda float64) []float64 {
	u := []float64{0.9}
	if s.trend {
		u = append(u, 1)
	}
	if s.damped {
		u = append(u, 3)
	}
	u = append(u, make([]float64, 2*len(s.periods)+s.p+s.q)...)
	if s.boxCox {
		l := min(max(lambda, 0.02), 0.98)
		u = append(u, math.Log(l/(1-l)))
	}
	return u
}

func (s tbatsShape) values(u []float64) tbatsValues {
	i := 0
	take := func() float64 {
		if i < len(u) {
			i++
			return u[i-1]
		}
		return 0
	}
	v := tbatsValues{phi: 1}
	v.alpha = 0.1 * take()
	if s.trend {
		v.beta = 0.05 * take()
	}
	if s.damped {
		v.phi = 0.8 + 0.2*logistic(take())
	}
	for range s.periods {
		g1 := 0.01 * take()
		v.gamma = append(v.gamma, [2]float64{g1, 0.01 * take()})
	}
	ar := make([]float64, s.p)
	for j := range ar {
		ar[j] = take()
	}
	ma := make([]float64, s.q)
	for j := range ma {
		ma[j] = take()
	}
	v.ar, v.ma = stationary(ar), negate(stationary(ma))
	if s.boxCox {
		v.lambda, v.hasLambda = logistic(take()), true
	}
	return v
}

type tbatsTurn struct {
	cos, sin float64
	period   int
}

// tbatsMachine is the model in innovations form: y(t) = w·x(t−1) + ε(t),
// x(t) = F x(t−1) + g ε(t).
type tbatsMachine struct {
	shape  tbatsShape
	values tbatsValues
	// (cos λ, sin λ) of each harmonic, in the order of the states
	turns []tbatsTurn
	w, g  []float64
}

func newTbatsMachine(s tbatsShape, v tbatsValues) tbatsMachine {
	n := s.states()
	m := tbatsMachine{shape: s, values: v, w: make([]float64, n), g: make([]float64, n)}
	m.w[0], m.g[0] = 1, v.alpha
	at := 1
	if s.trend {
		m.w[1], m.g[1] = v.phi, v.beta
		at = 2
	}
	for i, period := range s.periods {
		for j := 1; j <= s.harmonics[i]; j++ {
			angle := 2 * math.Pi * float64(j) / period
			m.turns = append(m.turns, tbatsTurn{math.Cos(angle), math.Sin(angle), i})
			m.w[at] = 1
			m.g[at], m.g[at+1] = v.gamma[i][0], v.gamma[i][1]
			at += 2
		}
	}
	copy(m.w[at:], v.ar)
	if s.p > 0 {
		m.g[at] = 1
	}
	at += s.p
	copy(m.w[at:], v.ma)
	if s.q > 0 {
		m.g[at] = 1
	}
	return m
}

// advance returns F x, using the structure of F.
func (m tbatsMachine) advance(x []float64) []float64 {
	s, v := m.shape, m.values
	seeds := s.seeds()
	// the part of the error process that is already known
	known := 0.0
	for i := range s.p + s.q {
		known += m.w[seeds+i] * x[seeds+i]
	}
	out := make([]float64, len(x))
	at := 1
	out[0] = x[0] + v.alpha*known
	if s.trend {
		out[0] += v.phi * x[1]
		out[1] = v.phi*x[1] + v.beta*known
		at = 2
	}
	for _, t := range m.turns {
		out[at] = t.cos*x[at] + t.sin*x[at+1] + v.gamma[t.period][0]*known
		out[at+1] = -t.sin*x[at] + t.cos*x[at+1] + v.gamma[t.period][1]*known
		at += 2
	}
	if s.p > 0 {
		out[at] = known
		for i := 1; i < s.p; i++ {
			out[at+i] = x[at+i-1]
		}
	}
	at += s.p
	for i := 1; i < s.q; i++ {
		out[at+i] = x[at+i-1]
	}
	return out
}

func (m tbatsMachine) observe(x []float64) float64 { return dot(m.w, x) }

// step returns the states after an error e.
func (m tbatsMachine) step(x []float64, e float64) []float64 {
	next := m.advance(x)
	for i, g := range m.g {
		next[i] += g * e
	}
	return next
}

// isStable reports whether the effect of an error does not grow: the matrix
// F − g w has no eigenvalue outside the unit circle, judged by how a vector
// grows when the matrix is applied to it many times. Eigenvalues on the
// circle are allowed (a seasonal pattern that never changes is one), and
// with them the slow growth of a vector that is not the fastest.
func (m tbatsMachine) isStable() bool {
	const rounds = 400
	x := make([]float64, m.shape.states())
	for i := range x {
		x[i] = 1 + 0.37*float64(i)
	}
	logGrowth := 0.0
	for round := range rounds {
		next := m.step(x, -m.observe(x))
		norm := math.Sqrt(dot(next, next))
		if !finite(norm) {
			return false
		}
		if norm == 0 {
			return true
		}
		// the first rounds carry the transient, not the long run
		if round >= rounds/2 {
			logGrowth += math.Log(norm)
		}
		for i := range next {
			next[i] /= norm
		}
		x = next
	}
	return logGrowth/float64(rounds/2) < 5e-3
}

// run runs the model over z from the states x; it returns the errors and the
// final states.
func (m tbatsMachine) run(z, x []float64) (errors, last []float64) {
	errors = make([]float64, len(z))
	for t, obs := range z {
		errors[t] = obs - m.observe(x)
		x = m.step(x, errors[t])
	}
	return errors, x
}

// seed returns the initial states that minimise the sum of squared errors.
// The errors are linear in them: ε(t) = ε̃(t) − r(t)·x₀, where ε̃ are the
// errors from zero states and r(t) follows its own recursion.
func (m tbatsMachine) seed(z []float64) ([]float64, bool) {
	seeds, states := m.shape.seeds(), m.shape.states()
	x := make([]float64, states)
	// column j: how the states depend on the j-th initial state
	columns := make([][]float64, seeds)
	for j := range columns {
		columns[j] = make([]float64, states)
		columns[j][j] = 1
	}
	gram := make([][]float64, seeds)
	for i := range gram {
		gram[i] = make([]float64, seeds)
	}
	moment := make([]float64, seeds)
	r := make([]float64, seeds)
	for _, obs := range z {
		e := obs - m.observe(x)
		for j, c := range columns {
			r[j] = m.observe(c)
		}
		for a := range seeds {
			moment[a] += r[a] * e
			for b := range seeds {
				gram[a][b] += r[a] * r[b]
			}
		}
		x = m.step(x, e)
		for j := range columns {
			columns[j] = m.step(columns[j], -r[j])
		}
	}
	// a whisper of ridge keeps the system solvable when two states are
	// nearly indistinguishable over the sample
	size := 0.0
	for a := range seeds {
		size = math.Max(size, gram[a][a])
	}
	for a := range seeds {
		gram[a][a] += 1e-12 * size
	}
	seed, ok := solve(gram, moment)
	if !ok {
		return nil, false
	}
	return append(seed, make([]float64, states-seeds)...), true
}

// tbatsAttempt is a fit of one shape.
type tbatsAttempt struct {
	shape      tbatsShape
	values     tbatsValues
	seed, last []float64
	errors     []float64
	likelihood float64
	aic        float64
}

func attemptTbats(shape tbatsShape, y []float64, startLambda float64, how effort) (tbatsAttempt, bool) {
	n := len(y)
	if n < shape.counted()+4 {
		return tbatsAttempt{}, false
	}
	logs := 0.0
	if shape.boxCox {
		for _, v := range y {
			logs += math.Log(v)
		}
	}
	evaluate := func(u []float64) (tbatsAttempt, bool) {
		values := shape.values(u)
		machine := newTbatsMachine(shape, values)
		if !machine.isStable() {
			return tbatsAttempt{}, false
		}
		z := y
		if values.hasLambda {
			z = make([]float64, n)
			t := BoxCox{values.lambda}
			for i, v := range y {
				z[i] = t.Apply(v)
			}
		}
		seed, ok := machine.seed(z)
		if !ok {
			return tbatsAttempt{}, false
		}
		errors, last := machine.run(z, seed)
		squares := dot(errors, errors)
		if !finite(squares) {
			return tbatsAttempt{}, false
		}
		for _, v := range last {
			if !finite(v) {
				return tbatsAttempt{}, false
			}
		}
		fit := -1e300
		if squares > 0 {
			fit = float64(n) * math.Log(squares)
		}
		lambda := 1.0
		if values.hasLambda {
			lambda = values.lambda
		}
		likelihood := fit - 2*(lambda-1)*logs
		return tbatsAttempt{
			shape: shape.clone(), values: values, seed: seed, last: last, errors: errors,
			likelihood: likelihood, aic: likelihood + 2*float64(shape.counted()),
		}, true
	}
	objective := func(u []float64) float64 {
		if a, ok := evaluate(u); ok {
			return a.likelihood
		}
		return inf
	}
	u, _ := nelderMead(objective, shape.start(startLambda), 0.5, how)
	return evaluate(u)
}

// tbatsCeilings returns the most harmonics each period can have: fewer than
// half the period, and none that another, shorter period already has.
func tbatsCeilings(periods []float64) []int {
	out := make([]int, len(periods))
	for i, p := range periods {
		most := int(math.Max(math.Floor((p-1)/2), 1))
		for _, shorter := range periods[:i] {
			ratio := p / shorter
			if math.Abs(ratio-math.Round(ratio)) < 1e-9 {
				most = min(most, max(int(math.Round(ratio))-1, 1))
			}
		}
		out[i] = most
	}
	return out
}

// TbatsFit is an estimated TBATS model.
type TbatsFit struct {
	attempt tbatsAttempt
	// Sigma2 is the variance of the errors on the scale of the fit.
	Sigma2 float64
}

// Select chooses what was left open and returns the estimated model.
func (t Tbats) Select(y Series) (*TbatsFit, error) {
	v := y.Values()
	if len(v) < 8 {
		return nil, ErrTooShort
	}
	if !y.IsFinite() {
		return nil, ErrNotFinite
	}
	var given []float64
	for _, p := range t.Periods {
		if finite(p) && p > 1 {
			given = append(given, p)
		}
	}
	slices.Sort(given)
	given = slices.Compact(given)
	if len(given) == 0 && y.Period() > 1 {
		given = []float64{float64(y.Period())}
	}
	var periods []float64
	for _, p := range given {
		if 2*p < float64(len(v)) {
			periods = append(periods, p)
		}
	}
	positive := y.IsPositive()
	if t.BoxCox == Yes && !positive {
		return nil, ErrNotPositive
	}
	lambda := 0.5
	if g, ok := Guerrero(y); ok {
		lambda = g.Lambda
	}
	ceilings := tbatsCeilings(periods)
	harmonics := make([]int, len(periods))
	for i := range harmonics {
		harmonics[i] = 1
		if t.Harmonics != nil {
			if i < len(t.Harmonics) {
				harmonics[i] = t.Harmonics[i]
			}
			harmonics[i] = min(max(harmonics[i], 1), ceilings[i])
		}
	}
	trend := t.Trend.or(true)
	best, ok := attemptTbats(tbatsShape{
		periods: periods, harmonics: harmonics, trend: trend,
		damped: trend && t.Damped.or(false), boxCox: t.BoxCox.or(false),
	}, v, lambda, quick)
	if !ok {
		return nil, ErrNoFit
	}
	consider := func(shape tbatsShape) bool {
		if a, ok := attemptTbats(shape, v, lambda, quick); ok && a.aic < best.aic {
			best = a
			return true
		}
		return false
	}

	// 1. harmonics of each period, one more for as long as it pays
	if t.Harmonics == nil {
		for i, most := range ceilings {
			for best.shape.harmonics[i] < most {
				shape := best.shape.clone()
				shape.harmonics[i]++
				if !consider(shape) {
					break
				}
			}
		}
	}
	// 2. transformation, trend and damping, each with and without ARMA
	//    errors of the orders that suit what it leaves unexplained
	transformations := []bool{false}
	switch {
	case t.BoxCox != Choose:
		transformations = []bool{t.BoxCox == Yes}
	case positive:
		transformations = []bool{false, true}
	}
	withArma := func(plain tbatsAttempt) (tbatsAttempt, bool) {
		if !t.ArmaErrors.or(true) && !t.FixOrders {
			return tbatsAttempt{}, false
		}
		p, q := t.P, t.Q
		if !t.FixOrders {
			auto, err := AutoArima{FixDifferences: true}.Select(NonSeasonal(plain.errors))
			if err != nil {
				return tbatsAttempt{}, false
			}
			p, _, q = auto.Order()
		}
		if p+q == 0 {
			return tbatsAttempt{}, false
		}
		shape := plain.shape.clone()
		shape.p, shape.q = p, q
		return attemptTbats(shape, v, lambda, thorough)
	}
	base := best.shape.clone()
	var winner *tbatsAttempt
	keep := func(a tbatsAttempt) {
		if winner == nil || a.aic < winner.aic {
			winner = &a
		}
	}
	for _, boxCox := range transformations {
		for _, c := range [][2]bool{{false, false}, {true, false}, {true, true}} {
			if !t.Trend.allows(c[0]) || (c[0] && !t.Damped.allows(c[1])) {
				continue
			}
			shape := base.clone()
			shape.boxCox, shape.trend, shape.damped = boxCox, c[0], c[1]
			plain, ok := attemptTbats(shape, v, lambda, thorough)
			if !ok {
				continue
			}
			if a, ok := withArma(plain); ok {
				keep(a)
			}
			if !t.FixOrders || t.P+t.Q == 0 {
				keep(plain)
			}
		}
	}
	if winner != nil && (winner.aic < best.aic || t.FixOrders) {
		best = *winner
	}
	return &TbatsFit{attempt: best, Sigma2: dot(best.errors, best.errors) / float64(len(v))}, nil
}

// SeasonalPeriod is a seasonal period of a TBATS model and the number of
// harmonics that describe its pattern.
type SeasonalPeriod struct {
	Period    float64
	Harmonics int
}

// Seasonal returns the seasonal periods and the harmonics of each.
func (f *TbatsFit) Seasonal() []SeasonalPeriod {
	s := f.attempt.shape
	out := make([]SeasonalPeriod, len(s.periods))
	for i := range out {
		out[i] = SeasonalPeriod{s.periods[i], s.harmonics[i]}
	}
	return out
}

// Lambda returns λ of the Box-Cox transformation, if one was used.
func (f *TbatsFit) Lambda() (float64, bool) {
	return f.attempt.values.lambda, f.attempt.values.hasLambda
}

// Trend returns the damping of the trend (1 for none), if there is a trend.
func (f *TbatsFit) Trend() (damping float64, ok bool) {
	return f.attempt.values.phi, f.attempt.shape.trend
}

// Arma returns the orders (p, q) of the ARMA errors.
func (f *TbatsFit) Arma() (p, q int) { return f.attempt.shape.p, f.attempt.shape.q }

// Smoothing returns the smoothing of the level and of the trend.
func (f *TbatsFit) Smoothing() (alpha, beta float64) {
	return f.attempt.values.alpha, f.attempt.values.beta
}

// Likelihood returns n ln Σε² − 2(λ − 1) Σ ln y: the likelihood as TBATS
// reports it, the lower the better.
func (f *TbatsFit) Likelihood() float64 { return f.attempt.likelihood }

// AIC returns the information criterion of the model.
func (f *TbatsFit) AIC() float64 { return f.attempt.aic }

// Residuals returns the errors of the one-step forecasts, on the scale of
// the fit.
func (f *TbatsFit) Residuals() []float64 { return f.attempt.errors }

// InitialStates returns the states before the first observation.
func (f *TbatsFit) InitialStates() []float64 { return f.attempt.seed }

func (f *TbatsFit) Forecast(h int) []float64 {
	a := f.attempt
	machine := newTbatsMachine(a.shape, a.values)
	x := a.last
	out := make([]float64, h)
	for k := range out {
		out[k] = machine.observe(x)
		x = machine.advance(x)
		if a.values.hasLambda {
			out[k] = BoxCox{a.values.lambda}.Invert(out[k])
		}
	}
	return out
}

func (f *TbatsFit) Params() []Param {
	a := f.attempt
	p := []Param{{"alpha", a.values.alpha}}
	if a.shape.trend {
		p = append(p, Param{"beta", a.values.beta})
	}
	if a.shape.damped {
		p = append(p, Param{"phi", a.values.phi})
	}
	if a.values.hasLambda {
		p = append(p, Param{"lambda", a.values.lambda})
	}
	for i, period := range a.shape.periods {
		k := strconv.Itoa(i + 1)
		p = append(p,
			Param{"period" + k, period}, Param{"harmonics" + k, float64(a.shape.harmonics[i])},
			Param{"gamma1_" + k, a.values.gamma[i][0]}, Param{"gamma2_" + k, a.values.gamma[i][1]})
	}
	for i, c := range a.values.ar {
		p = append(p, Param{"ar" + strconv.Itoa(i+1), c})
	}
	for i, c := range a.values.ma {
		p = append(p, Param{"ma" + strconv.Itoa(i+1), c})
	}
	return append(p, Param{"sigma2", f.Sigma2}, Param{"aic", a.aic})
}

func (Tbats) Name() string { return "tbats" }
func (Tbats) Description() string {
	return "TBATS: trigonometric seasonality, Box-Cox, ARMA errors and trend, chosen by AIC"
}

func (t Tbats) Fit(y Series) (Fitted, error) {
	fit, err := t.Select(y)
	if err != nil {
		return nil, err
	}
	return fit, nil
}
