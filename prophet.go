package foresight

import "math"

// Prophet is the Prophet model (Taylor & Letham, 2018): a piecewise linear
// trend that may bend at changepoints, a Fourier seasonal pattern and dated
// events, fitted by maximum a posteriori.
//
// y(t) = g(t) + s(t) + events(t) + noise, where g is a line whose slope may
// change at each of a set of candidate changepoints spread over the first
// part of the history. The changes have a Laplace prior, so most of them
// come out as exactly zero and the trend bends only where the data insists;
// the seasonal and event coefficients have normal priors.
//
// What is and is not here, compared with the original: observations are
// equally spaced and the seasonal period is the one of the series; growth is
// linear and the components are additive (for multiplicative behaviour, fit
// on the log scale with [Log]); events are given by position instead of by
// calendar. The estimate is the exact optimum of the same posterior, found
// without Stan.
//
// The zero value has the defaults of Prophet; see [DefaultProphet].
type Prophet struct {
	// Changepoints is the number of candidate changepoints (default 25).
	// Use NoChangepoints for a straight line.
	Changepoints int
	// NoChangepoints fits a straight line.
	NoChangepoints bool
	// ChangepointRange is the share of the history, from the start, where
	// changepoints may fall (default 0.8).
	ChangepointRange float64
	// ChangepointPriorScale is the flexibility of the trend: larger values
	// let it bend more (default 0.05).
	ChangepointPriorScale float64
	// SeasonalityPriorScale is the flexibility of the seasonal pattern
	// (default 10).
	SeasonalityPriorScale float64
	// FourierOrder is the number of harmonics of the seasonal pattern
	// (default 10, or half the period when that is less, which describes any
	// pattern of that period). Use NoSeasonality for none. There is no
	// seasonal pattern before two full cycles of history either.
	FourierOrder int
	// NoSeasonality leaves the seasonal pattern out.
	NoSeasonality bool
	// EventPriorScale is the flexibility of the effects of events
	// (default 10).
	EventPriorScale float64
	// Events are recurring events and lasting changes of level.
	Events []Event
}

// Event is something dated that moves the series.
type Event struct {
	// Name identifies the effect of the event among the parameters.
	Name string
	// Positions are where a recurring event happens, in the ORIGINAL data
	// (see [Series.Index]), past and future: one effect is estimated for all
	// its occurrences. Positions beyond the history are how the forecast
	// knows the event is coming.
	Positions []int
	// Step makes the event a lasting change of level, from StepFrom on, such
	// as a change in the law.
	Step     bool
	StepFrom int
}

func (e Event) at(index int) float64 {
	if e.Step {
		if index >= e.StepFrom {
			return 1
		}
		return 0
	}
	for _, p := range e.Positions {
		if p == index {
			return 1
		}
	}
	return 0
}

// DefaultProphet returns the defaults of Prophet spelled out: 25 candidate
// changepoints over the first 80% of the history with prior scale 0.05,
// seasonal and event prior scales of 10.
func DefaultProphet() Prophet {
	return Prophet{
		Changepoints: 25, ChangepointRange: 0.8, ChangepointPriorScale: 0.05,
		SeasonalityPriorScale: 10, FourierOrder: 10, EventPriorScale: 10,
	}
}

func (p Prophet) filled() Prophet {
	d := DefaultProphet()
	if p.Changepoints <= 0 {
		p.Changepoints = d.Changepoints
	}
	if p.NoChangepoints {
		p.Changepoints = 0
	}
	if p.ChangepointRange == 0 {
		p.ChangepointRange = d.ChangepointRange
	}
	p.ChangepointRange = min(max(p.ChangepointRange, 0), 1)
	if p.ChangepointPriorScale == 0 {
		p.ChangepointPriorScale = d.ChangepointPriorScale
	}
	if p.SeasonalityPriorScale == 0 {
		p.SeasonalityPriorScale = d.SeasonalityPriorScale
	}
	if p.FourierOrder <= 0 {
		p.FourierOrder = d.FourierOrder
	}
	if p.NoSeasonality {
		p.FourierOrder = 0
	}
	if p.EventPriorScale == 0 {
		p.EventPriorScale = d.EventPriorScale
	}
	return p
}

// harmonics actually used for n observations of a series of period m: none
// before two full cycles, as a pattern seen once cannot be told from the
// trend.
func (p Prophet) harmonics(m, n int) int {
	if m < 2 || n < 2*m {
		return 0
	}
	return min(p.FourierOrder, m/2)
}

// prophetLayout says where each group of coefficients sits and how to build
// a row of the design.
type prophetLayout struct {
	// n − 1: positions are divided by it, so the history spans [0, 1]
	span float64
	// changepoints on that scale
	changepoints []float64
	harmonics    int
	period       int
	events       []Event
}

// seasonal columns: a sine and a cosine per harmonic, except that the sine
// of the harmonic at half the period is zero everywhere.
func (l prophetLayout) seasonal() int {
	full := 2 * l.harmonics
	if l.harmonics > 0 && 2*l.harmonics == l.period {
		return full - 1
	}
	return full
}

func (l prophetLayout) columns() int {
	return 2 + len(l.changepoints) + l.seasonal() + len(l.events)
}

func (l prophetLayout) rangeChangepoints() (int, int) { return 2, 2 + len(l.changepoints) }

func (l prophetLayout) rangeSeasonal() (int, int) {
	first := 2 + len(l.changepoints)
	return first, first + l.seasonal()
}

func (l prophetLayout) rangeEvents() (int, int) {
	first := 2 + len(l.changepoints) + l.seasonal()
	return first, first + len(l.events)
}

// row of the design for the observation at relative position i, whose
// position in the original data is index and whose season is season.
func (l prophetLayout) row(i, index, season int) []float64 {
	t := float64(i) / l.span
	row := make([]float64, 0, l.columns())
	row = append(row, t, 1)
	for _, s := range l.changepoints {
		row = append(row, math.Max(t-s, 0))
	}
	for h := 1; h <= l.harmonics; h++ {
		angle := 2 * math.Pi * float64(h*season) / float64(l.period)
		if 2*h != l.period {
			row = append(row, math.Sin(angle))
		}
		row = append(row, math.Cos(angle))
	}
	for _, e := range l.events {
		row = append(row, e.at(index))
	}
	return row
}

func signum(x float64) float64 { return math.Copysign(1, x) }

// penalised is weight/2 · ‖y − Aθ‖² + Σ ridgeⱼ θⱼ²/2 + Σ lassoⱼ |θⱼ|, in
// terms of AᵀA and Aᵀy.
type penalised struct {
	gram         [][]float64
	moment       []float64
	ridge, lasso []float64
}

// activeSet finds the exact minimum by an active-set method: solve for the
// coefficients that are in play, stop at the first one that would change
// sign and drop it, bring in the one the optimality conditions complain most
// about, until nothing moves. It reports false if it gives up (singular
// system or too many rounds), leaving theta at the best point reached.
func (q penalised) activeSet(weight float64, theta []float64) bool {
	p := len(theta)
	usable := make([]bool, p)
	sign := make([]float64, p)
	active := make([]bool, p)
	for j := range p {
		usable[j] = q.gram[j][j] > 0
		if theta[j] != 0 {
			sign[j] = signum(theta[j])
		}
		active[j] = usable[j] && (q.lasso[j] == 0 || theta[j] != 0)
	}
	for range 20*p + 50 {
		var members []int
		for j := range p {
			if active[j] {
				members = append(members, j)
			}
		}
		a := make([][]float64, len(members))
		b := make([]float64, len(members))
		for r, i := range members {
			a[r] = make([]float64, len(members))
			for c, j := range members {
				a[r][c] = weight * q.gram[i][j]
				if i == j {
					a[r][c] += q.ridge[i]
				}
			}
			b[r] = weight*q.moment[i] - q.lasso[i]*sign[i]
		}
		solution, ok := solve(a, b)
		if !ok {
			return false
		}
		// as far towards the solution as no coefficient changes sign
		step, stopped := 1.0, -1
		for k, j := range members {
			if q.lasso[j] > 0 && solution[k]*sign[j] < 0 {
				if reach := theta[j] / (theta[j] - solution[k]); reach < step {
					step, stopped = reach, j
				}
			}
		}
		for k, j := range members {
			theta[j] += step * (solution[k] - theta[j])
		}
		if stopped >= 0 {
			theta[stopped], sign[stopped], active[stopped] = 0, 0, false
			continue
		}
		// is any coefficient left at zero pulled harder than its penalty?
		worst, worstExcess, direction := -1, 0.0, 0.0
		for j := range p {
			if !usable[j] || active[j] {
				continue
			}
			pull := weight * (q.moment[j] - dot(q.gram[j], theta))
			excess := math.Abs(pull) - q.lasso[j]*(1+1e-9) - 1e-12
			if excess > 0 && (worst < 0 || excess > worstExcess) {
				worst, worstExcess, direction = j, excess, signum(pull)
			}
		}
		if worst < 0 {
			return true
		}
		active[worst], sign[worst] = true, direction
	}
	return false
}

// descend finds the same minimum by coordinate descent: slower, never stuck.
func (q penalised) descend(weight float64, theta []float64) {
	p := len(theta)
	// Aᵀ(y − Aθ), kept up to date as θ moves
	pull := make([]float64, p)
	for j := range pull {
		pull[j] = q.moment[j] - dot(q.gram[j], theta)
	}
	for range 50_000 {
		moved := 0.0
		for j := range p {
			norm := q.gram[j][j]
			if norm == 0 {
				continue
			}
			old := theta[j]
			force := weight * (pull[j] + norm*old)
			shrunk := signum(force) * math.Max(math.Abs(force)-q.lasso[j], 0)
			next := shrunk / (weight*norm + q.ridge[j])
			if next != old {
				for i := range pull {
					pull[i] -= (next - old) * q.gram[j][i]
				}
				theta[j] = next
				moved = math.Max(moved, math.Abs(next-old)*math.Sqrt(norm))
			}
		}
		if moved < 1e-12 {
			break
		}
	}
}

// ProphetFit is an estimated Prophet model.
type ProphetFit struct {
	layout prophetLayout
	// coefficients on the scale of the fit (values divided by scale)
	theta       []float64
	sigma       float64
	scale       float64
	n           int
	start       int
	firstSeason int
	at          []int
}

// empty reports whether the fit is the zero value rather than an estimate.
func (f *ProphetFit) empty() bool { return f == nil || len(f.theta) == 0 }

// Estimate fits the model and returns everything that was estimated.
func (p Prophet) Estimate(y Series) (*ProphetFit, error) {
	p = p.filled()
	v, n, m := y.Values(), y.Len(), y.Period()
	if n < 4 {
		return nil, ErrTooShort
	}
	if !y.IsFinite() {
		return nil, ErrNotFinite
	}
	for _, s := range []float64{p.ChangepointPriorScale, p.SeasonalityPriorScale, p.EventPriorScale} {
		if !finite(s) || s <= 0 {
			return nil, ErrNoFit
		}
	}
	if math.IsNaN(p.ChangepointRange) {
		return nil, ErrConfig
	}
	scale := 0.0
	for _, x := range v {
		scale = math.Max(scale, math.Abs(x))
	}
	if scale <= 0 {
		scale = 1
	}
	target := make([]float64, n)
	for i, x := range v {
		target[i] = x / scale
	}
	span := float64(n - 1)

	// candidate changepoints, evenly spread over the first part of the history
	history := max(int(math.Floor(float64(n)*p.ChangepointRange)), 1)
	candidates := min(p.Changepoints, history-1)
	var at []int
	for i := 1; i <= candidates; i++ {
		k := int(math.RoundToEven(float64(i) * float64(history-1) / float64(candidates)))
		if len(at) == 0 || at[len(at)-1] != k {
			at = append(at, k)
		}
	}
	layout := prophetLayout{span: span, harmonics: p.harmonics(m, n), period: m, events: p.Events}
	for _, i := range at {
		layout.changepoints = append(layout.changepoints, float64(i)/span)
	}
	columns := layout.columns()
	// penalties of each column: ridge (1/variance) and lasso (1/scale)
	ridge := make([]float64, columns)
	lasso := make([]float64, columns)
	ridge[0], ridge[1] = 1.0/25, 1.0/25
	for j, to := layout.rangeChangepoints(); j < to; j++ {
		lasso[j] = 1 / p.ChangepointPriorScale
	}
	for j, to := layout.rangeSeasonal(); j < to; j++ {
		ridge[j] = 1 / (p.SeasonalityPriorScale * p.SeasonalityPriorScale)
	}
	for j, to := layout.rangeEvents(); j < to; j++ {
		ridge[j] = 1 / (p.EventPriorScale * p.EventPriorScale)
	}

	// the design matrix, by row and by column
	rows := make([][]float64, n)
	for i := range rows {
		rows[i] = layout.row(i, y.Index(i), y.Season(i))
	}
	design := make([][]float64, columns)
	for j := range design {
		design[j] = make([]float64, n)
		for i, r := range rows {
			design[j][i] = r[j]
		}
	}
	// everything the search needs from the data: AᵀA and Aᵀy
	problem := penalised{gram: make([][]float64, columns), moment: make([]float64, columns), ridge: ridge, lasso: lasso}
	for i, a := range design {
		problem.gram[i] = make([]float64, columns)
		for j, b := range design {
			problem.gram[i][j] = dot(a, b)
		}
		problem.moment[i] = dot(a, target)
	}

	// start: the line through the first and last observations
	theta := make([]float64, columns)
	theta[0] = target[n-1] - target[0]
	theta[1] = target[0]
	sigma := 1.0
	for range 500 {
		// given the noise level, a lasso with ridge terms
		weight := 1 / (sigma * sigma)
		if !problem.activeSet(weight, theta) {
			problem.descend(weight, theta)
		}
		// given the fit, the noise level in closed form
		rss := 0.0
		for i, row := range rows {
			d := target[i] - dot(row, theta)
			rss += d * d
		}
		nf := float64(n)
		next := math.Max(math.Sqrt((math.Sqrt(nf*nf+16*rss)-nf)/8), 1e-8)
		settled := math.Abs(next-sigma) <= 1e-10*sigma
		sigma = next
		if settled {
			break
		}
	}
	for _, x := range theta {
		if !finite(x) {
			return nil, ErrNoFit
		}
	}
	return &ProphetFit{
		layout: layout, theta: theta, sigma: sigma, scale: scale, n: n,
		start: y.Start(), firstSeason: y.Season(0), at: at,
	}, nil
}

// components are not numbers for a fit that did not come from
// [Prophet.Estimate].
func (f *ProphetFit) components(i int) (trend, seasonal, events float64) {
	if f.empty() {
		return math.NaN(), math.NaN(), math.NaN()
	}
	season := (f.firstSeason + i) % max(f.layout.period, 1)
	row := f.layout.row(i, f.start+i, season)
	part := func(from, to int) float64 {
		s := 0.0
		for j := from; j < to; j++ {
			s += row[j] * f.theta[j]
		}
		return s * f.scale
	}
	a, b := f.layout.rangeChangepoints()
	trend = part(0, 2) + part(a, b)
	a, b = f.layout.rangeSeasonal()
	seasonal = part(a, b)
	a, b = f.layout.rangeEvents()
	return trend, seasonal, part(a, b)
}

// Trend returns the trend at the relative position i (0 is the first
// observation; the length of the history and beyond are the future).
func (f *ProphetFit) Trend(i int) float64 { t, _, _ := f.components(i); return t }

// Seasonal returns the seasonal effect at the relative position i.
func (f *ProphetFit) Seasonal(i int) float64 { _, s, _ := f.components(i); return s }

// Events returns the effect of the events at the relative position i.
func (f *ProphetFit) Events(i int) float64 { _, _, e := f.components(i); return e }

// Changepoint is a point where the trend did bend.
type Changepoint struct {
	// Position is relative to the first observation.
	Position int
	// Change is the change in slope, in units of the series per period.
	Change float64
}

// Changepoints returns the changepoints where the trend did bend.
func (f *ProphetFit) Changepoints() []Changepoint {
	if f.empty() {
		return nil
	}
	var out []Changepoint
	first, _ := f.layout.rangeChangepoints()
	for k, i := range f.at {
		if delta := f.theta[first+k]; delta != 0 {
			out = append(out, Changepoint{i, delta * f.scale / f.layout.span})
		}
	}
	return out
}

// Effects returns the estimated effect of each event, in units of the
// series.
func (f *ProphetFit) Effects() []Param {
	if f.empty() {
		return nil
	}
	first, _ := f.layout.rangeEvents()
	out := make([]Param, len(f.layout.events))
	for k, e := range f.layout.events {
		out[k] = Param{e.Name, f.theta[first+k] * f.scale}
	}
	return out
}

func (f *ProphetFit) values(from, to int) []float64 {
	out := make([]float64, 0, to-from)
	for i := from; i < to; i++ {
		t, s, e := f.components(i)
		out = append(out, t+s+e)
	}
	return out
}

// FittedValues returns the values fitted to the history.
func (f *ProphetFit) FittedValues() []float64 {
	if f.empty() {
		return nil
	}
	return f.values(0, f.n)
}

// Sigma returns the standard deviation of the noise, in units of the series.
func (f *ProphetFit) Sigma() float64 {
	if f.empty() {
		return math.NaN()
	}
	return f.sigma * f.scale
}

// Forecast returns the forecasts for the h periods after the last
// observation; nothing for an h of zero or less, or for a fit that did not
// come from [Prophet.Estimate].
func (f *ProphetFit) Forecast(h int) []float64 {
	if h <= 0 || f.empty() {
		return nil
	}
	return f.values(f.n, f.n+h)
}

// Params returns the slopes of the trend, the number of changepoints, the
// noise and the effects of the events; nothing for a fit that did not come
// from [Prophet.Estimate].
func (f *ProphetFit) Params() []Param {
	if f.empty() {
		return nil
	}
	slope := f.scale / f.layout.span
	final := f.theta[0]
	for j, to := f.layout.rangeChangepoints(); j < to; j++ {
		final += f.theta[j]
	}
	p := []Param{
		{"initial_slope", f.theta[0] * slope},
		{"final_slope", final * slope},
		{"changepoints", float64(len(f.Changepoints()))},
		{"sigma", f.Sigma()},
	}
	for _, e := range f.Effects() {
		p = append(p, Param{"event_" + e.Name, e.Value})
	}
	return p
}

// Name is the identifier of the model.
func (Prophet) Name() string { return "prophet" }

// Description is a one-line description of the model.
func (Prophet) Description() string {
	return "Prophet: piecewise linear trend with changepoints, Fourier seasonality and events"
}

// Fit estimates the model; see [Prophet.Estimate].
func (p Prophet) Fit(y Series) (Fitted, error) {
	fit, err := p.Estimate(y)
	if err != nil {
		return nil, err
	}
	return fit, nil
}
