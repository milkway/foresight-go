package foresight

import (
	"errors"
	"math"
	"strconv"
)

// Constant says whether an ARIMA model estimates a constant: the mean when
// the series is not differenced, the drift when it is differenced once. With
// two or more differences there is never a constant.
type Constant int

const (
	// ConstantByDefault estimates the mean of a series that is not
	// differenced and nothing otherwise.
	ConstantByDefault Constant = iota
	// WithConstant estimates the mean or the drift.
	WithConstant
	// WithoutConstant estimates neither.
	WithoutConstant
)

// Arima is ARIMA(P, D, Q)(SeasonalP, SeasonalD, SeasonalQ) with the seasonal
// period of the series.
//
// The series is differenced (D ordinary and SeasonalD seasonal differences)
// and an ARMA model is fitted to the result by exact Gaussian maximum
// likelihood, computed with the innovations algorithm. The search runs over
// partial autocorrelations, so the estimates are always stationary and
// invertible. The likelihood of a mixed model can have more than one peak, so
// the search starts from three points and keeps the best.
//
// With Regressors, the model is a regression whose errors follow the ARIMA
// model; the coefficients of the regression and of the model are estimated
// together.
//
// For series whose swings grow with the level, fit on the log scale with
// [Log].
type Arima struct {
	P, D, Q int
	// Seasonal orders, ignored for series without seasonality.
	SeasonalP, SeasonalD, SeasonalQ int
	Constant                        Constant
	// Regressors are external variables, with rows for the history and for
	// the periods to forecast. The zero value is none.
	Regressors Regressors
}

// Airline returns the "airline model", ARIMA(0,1,1)(0,1,1): a good default
// for seasonal series, usually on the log scale.
func Airline() Arima {
	return Arima{D: 1, Q: 1, SeasonalD: 1, SeasonalQ: 1}
}

// ErrRegressors is returned when the regressors do not cover the history or
// the horizon, or cannot be told apart from each other or from the constant.
var ErrRegressors = errors.New("foresight: regressors do not cover the series or are collinear")

func (a Arima) differences() int { return a.D + a.SeasonalD }

func (a Arima) hasConstant() bool {
	switch a.differences() {
	case 0:
		return a.Constant != WithoutConstant
	case 1:
		return a.Constant == WithConstant
	}
	return false
}

// forPeriod is the model without the seasonal part, for series of period 1.
func (a Arima) forPeriod(m int) Arima {
	if m < 2 {
		a.SeasonalP, a.SeasonalD, a.SeasonalQ = 0, 0, 0
	}
	return a
}

func repeat(v float64, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// start returns unconstrained starting values with the autoregressive
// partial autocorrelations at ar and the moving average ones at ma.
func (a Arima) start(ar, ma float64) []float64 {
	var u []float64
	u = append(u, repeat(math.Atanh(ar), a.P)...)
	u = append(u, repeat(math.Atanh(ma), a.Q)...)
	u = append(u, repeat(math.Atanh(ar), a.SeasonalP)...)
	u = append(u, repeat(math.Atanh(ma), a.SeasonalQ)...)
	return u
}

// startNear returns starting values taken from a fitted model of other
// orders: what both have in common is kept and the rest starts at zero,
// which describes the same process.
func (a Arima) startNear(other *ArimaFit) []float64 {
	o := other.spec
	blocks := [][2]int{{a.P, o.P}, {a.Q, o.Q}, {a.SeasonalP, o.SeasonalP}, {a.SeasonalQ, o.SeasonalQ}}
	var u []float64
	at := 0
	for _, b := range blocks {
		for i := range b[0] {
			if i < b[1] {
				u = append(u, other.unconstrained[at+i])
			} else {
				u = append(u, 0)
			}
		}
		at += b[1]
	}
	return u
}

// differencing returns the coefficients of (1 − B)ᵈ (1 − Bᵐ)ᴰ, from the
// power 0.
func differencing(d, sd, m int) []float64 {
	c := []float64{1}
	times := func(lag int) {
		next := make([]float64, len(c)+lag)
		copy(next, c)
		for i, v := range c {
			next[i+lag] -= v
		}
		c = next
	}
	for range d {
		times(1)
	}
	for range sd {
		times(m)
	}
	return c
}

type arimaParts struct{ ar, ma, sar, sma []float64 }

// arma multiplies out the seasonal and non-seasonal polynomials.
func (p arimaParts) arma(m int) arma {
	return arma{
		phi:   negate(expand(negate(p.ar), negate(p.sar), m)),
		theta: expand(p.ma, p.sma, m),
	}
}

// evaluation is the likelihood at one point of the search and what comes
// with it.
type evaluation struct {
	sigma2, logDet float64
	// coefficients of the constant and of the regressors
	beta []float64
	// the differenced series without what the regression explains
	centred []float64
	errors  []float64
}

// ArimaFit is an estimated ARIMA model.
type ArimaFit struct {
	spec   Arima
	period int
	arma   arma
	// AR holds φ₁, φ₂, …: X(t) = φ₁X(t−1) + … + Z(t) + θ₁Z(t−1) + …
	AR []float64
	// MA holds θ₁, θ₂, …
	MA []float64
	// SeasonalAR holds Φ₁, Φ₂, … at the seasonal lags.
	SeasonalAR []float64
	// SeasonalMA holds Θ₁, Θ₂, … at the seasonal lags.
	SeasonalMA []float64
	// HasConstant says whether a constant was estimated; Constant is then
	// the mean of the differenced series.
	HasConstant bool
	Constant    float64
	// Regression has the coefficient of each regressor.
	Regression []Param
	// Sigma2 is the innovation variance (maximum likelihood, not corrected
	// for degrees of freedom).
	Sigma2        float64
	LogLikelihood float64
	AIC           float64
	AICc          float64
	BIC           float64
	// Residuals are the one-step prediction errors of the differenced
	// series.
	Residuals     []float64
	centred       []float64
	unconstrained []float64
	// last observations of the original series, to undo the differences
	tail  []float64
	delta []float64
	// position in the original data of the first period to forecast
	next int
}

// Estimate fits the model and returns everything that was estimated.
func (a Arima) Estimate(y Series) (*ArimaFit, error) {
	return a.estimateFrom(y, nil)
}

// estimateFrom is Estimate, or a quicker search from a neighbouring model
// when one is given, which is enough to compare models.
func (a Arima) estimateFrom(y Series, near *ArimaFit) (*ArimaFit, error) {
	m := y.Period()
	spec := a.forPeriod(m)
	if !y.IsFinite() {
		return nil, ErrNotFinite
	}
	delta := differencing(spec.D, spec.SeasonalD, m)
	lost := len(delta) - 1
	v := y.Values()
	if len(v) <= lost {
		return nil, ErrTooShort
	}
	differenced := func(level func(int) float64) []float64 {
		out := make([]float64, 0, len(v)-lost)
		for t := lost; t < len(v); t++ {
			s := 0.0
			for k, c := range delta {
				s += c * level(t-k)
			}
			out = append(out, s)
		}
		return out
	}
	w := differenced(func(t int) float64 { return v[t] })
	n := len(w)
	coefficients := spec.P + spec.Q + spec.SeasonalP + spec.SeasonalQ
	longest := max(spec.P+spec.SeasonalP*m, spec.Q+spec.SeasonalQ*m)
	constant := spec.hasConstant()
	// what is explained by regression: the constant and the regressors, on
	// the differenced scale
	var explained [][]float64
	if constant {
		explained = append(explained, repeat(1, n))
	}
	if spec.Regressors.Width() > 0 {
		if !spec.Regressors.Covers(y.Index(len(v))) {
			return nil, ErrRegressors
		}
		for _, column := range spec.Regressors.columns {
			explained = append(explained, differenced(func(t int) float64 { return column[y.Index(t)] }))
		}
	}
	width := len(explained)
	if n < longest+coefficients+width+8 {
		return nil, ErrTooShort
	}

	build := func(u []float64) arimaParts {
		ar, rest := u[:spec.P], u[spec.P:]
		ma, rest := rest[:spec.Q], rest[spec.Q:]
		sar, sma := rest[:spec.SeasonalP], rest[spec.SeasonalP:]
		return arimaParts{
			ar: stationary(ar), ma: negate(stationary(ma)),
			sar: stationary(sar), sma: negate(stationary(sma)),
		}
	}
	// for given ARMA coefficients the prediction errors are linear in the
	// data, so the regression is least squares on the filtered columns
	evaluate := func(u []float64) (evaluation, bool) {
		process := build(u).arma(m)
		inn, ok := process.innovate(n - 1)
		if !ok {
			return evaluation{}, false
		}
		scaled := func(x []float64) []float64 {
			e := process.filter(x, inn).errors
			for i := range e {
				e[i] /= math.Sqrt(inn.v[i])
			}
			return e
		}
		target := scaled(w)
		var beta []float64
		if width > 0 {
			columns := make([][]float64, width)
			for j, c := range explained {
				columns[j] = scaled(c)
			}
			gram := make([][]float64, width)
			moment := make([]float64, width)
			for i, ci := range columns {
				gram[i] = make([]float64, width)
				for j, cj := range columns {
					gram[i][j] = dot(ci, cj)
				}
				moment[i] = dot(ci, target)
			}
			if beta, ok = solve(gram, moment); !ok {
				return evaluation{}, false
			}
		}
		centred := make([]float64, n)
		for t := range centred {
			fit := 0.0
			for j := range width {
				fit += beta[j] * explained[j][t]
			}
			centred[t] = w[t] - fit
		}
		f := process.filter(centred, inn)
		sigma2 := f.sumSquares / float64(n)
		if !finite(sigma2) || !finite(f.logDet) {
			return evaluation{}, false
		}
		return evaluation{sigma2, f.logDet, beta, centred, f.errors}, true
	}
	// −2 log likelihood with the variance concentrated out, up to a constant
	objective := func(u []float64) float64 {
		e, ok := evaluate(u)
		switch {
		case !ok:
			return math.Inf(1)
		case e.sigma2 > 0:
			return float64(n)*math.Log(e.sigma2) + e.logDet
		}
		// a perfect fit: nothing left to explain
		return -1e300
	}

	starts, how := [][]float64{spec.start(0, 0)}, thorough
	switch {
	case near != nil:
		starts, how = [][]float64{spec.startNear(near)}, quick
	case coefficients > 1:
		starts = append(starts, spec.start(0.5, 0.5), spec.start(-0.5, -0.5))
	}
	var u []float64
	best := math.NaN()
	for i, s := range starts {
		if x, value := nelderMead(objective, s, 0.5, how); i == 0 || below(value, best) {
			u, best = x, value
		}
	}
	e, ok := evaluate(u)
	if !ok {
		return nil, ErrNoFit
	}
	parts := build(u)
	nf := float64(n)
	k := float64(coefficients + width + 1)
	logLikelihood := math.Inf(1)
	if e.sigma2 > 0 {
		logLikelihood = -0.5 * (nf*math.Log(2*math.Pi*e.sigma2) + e.logDet + nf)
	}
	aic := -2*logLikelihood + 2*k
	aicc := math.Inf(1)
	if nf-k-1 > 0 {
		aicc = aic + 2*k*(k+1)/(nf-k-1)
	}
	fit := &ArimaFit{
		spec: spec, period: m, arma: parts.arma(m),
		AR: parts.ar, MA: parts.ma, SeasonalAR: parts.sar, SeasonalMA: parts.sma,
		HasConstant: constant,
		Sigma2:      e.sigma2, LogLikelihood: logLikelihood,
		AIC: aic, AICc: aicc, BIC: -2*logLikelihood + k*math.Log(nf),
		Residuals: e.errors, centred: e.centred, unconstrained: u,
		tail: append([]float64(nil), v[len(v)-lost:]...), delta: delta,
		next: y.Index(len(v)),
	}
	slopes := e.beta
	if constant {
		fit.Constant, slopes = e.beta[0], e.beta[1:]
	}
	for j, name := range spec.Regressors.names {
		fit.Regression = append(fit.Regression, Param{name, slopes[j]})
	}
	return fit, nil
}

// Order returns the orders (p, d, q).
func (f *ArimaFit) Order() (p, d, q int) { return f.spec.P, f.spec.D, f.spec.Q }

// SeasonalOrder returns the seasonal orders (P, D, Q).
func (f *ArimaFit) SeasonalOrder() (p, d, q int) {
	return f.spec.SeasonalP, f.spec.SeasonalD, f.spec.SeasonalQ
}

// Period returns the seasonal period of the series the model was fitted on.
func (f *ArimaFit) Period() int { return f.period }

// IsWellBehaved reports whether every root of the four polynomials is
// farther from the unit circle than margin (such as 1.01).
func (f *ArimaFit) IsWellBehaved(margin float64) bool {
	return rootsOutside(f.AR, margin) && rootsOutside(f.SeasonalAR, margin) &&
		rootsOutside(negate(f.MA), margin) && rootsOutside(negate(f.SeasonalMA), margin)
}

// ForecastVariance returns the variance of the forecast error 1 to h periods
// ahead, on the scale the model was fitted on.
func (f *ArimaFit) ForecastVariance(h int) []float64 {
	// ψ weights of the model with the differences put back
	ar := append([]float64{1}, negate(f.arma.phi)...)
	full := make([]float64, len(ar)+len(f.delta)-1)
	for i, a := range ar {
		for j, b := range f.delta {
			full[i+j] += a * b
		}
	}
	psi := arma{phi: negate(full[1:]), theta: f.arma.theta}.psi(max(h-1, 0))
	out := make([]float64, h)
	total := 0.0
	for k := range out {
		total += psi[k] * psi[k]
		out[k] = total * f.Sigma2
	}
	return out
}

// Forecast returns the forecasts for the h periods after the last
// observation. Where the regressors have no value for a period, the forecast
// is not a number.
func (f *ArimaFit) Forecast(h int) []float64 {
	n := len(f.centred)
	differenced := make([]float64, h)
	if inn, ok := f.arma.innovate(n + max(h-1, 0)); ok {
		differenced = f.arma.forecast(f.centred, f.Residuals, inn, h)
	}
	lost := len(f.delta) - 1
	// what the regressors explain of the differenced series at a position of
	// the original data
	explained := func(at int) float64 {
		total := 0.0
		for j, column := range f.spec.Regressors.columns {
			change := 0.0
			for k, c := range f.delta {
				if at-k < len(column) {
					change += c * column[at-k]
				} else {
					change += math.NaN()
				}
			}
			total += f.Regression[j].Value * change
		}
		return total
	}
	// undo the differences: y(t) = w(t) − Σ δₖ y(t − k)
	path := append([]float64(nil), f.tail...)
	for k, w := range differenced {
		back := 0.0
		for i := 1; i <= lost; i++ {
			back += f.delta[i] * path[len(path)-i]
		}
		path = append(path, w+f.Constant+explained(f.next+k)-back)
	}
	return path[lost:]
}

// Params returns the coefficients, the orders and the fit.
func (f *ArimaFit) Params() []Param {
	var p []Param
	push := func(prefix string, values []float64) {
		for i, v := range values {
			p = append(p, Param{prefix + strconv.Itoa(i+1), v})
		}
	}
	push("ar", f.AR)
	push("ma", f.MA)
	push("sar", f.SeasonalAR)
	push("sma", f.SeasonalMA)
	if f.HasConstant {
		// per period of the original series
		name := "drift"
		if f.spec.differences() == 0 {
			name = "mean"
		}
		scale := math.Pow(float64(f.period), float64(f.spec.SeasonalD))
		p = append(p, Param{name, f.Constant / scale})
	}
	for _, r := range f.Regression {
		p = append(p, Param{"x_" + r.Name, r.Value})
	}
	p = append(p,
		Param{"p", float64(f.spec.P)}, Param{"d", float64(f.spec.D)}, Param{"q", float64(f.spec.Q)},
		Param{"P", float64(f.spec.SeasonalP)}, Param{"D", float64(f.spec.SeasonalD)}, Param{"Q", float64(f.spec.SeasonalQ)},
		Param{"sigma2", f.Sigma2}, Param{"log_likelihood", f.LogLikelihood}, Param{"aicc", f.AICc},
	)
	return p
}

func (a Arima) Name() string {
	name := "arima_" + strconv.Itoa(a.P) + strconv.Itoa(a.D) + strconv.Itoa(a.Q)
	if a.SeasonalP+a.SeasonalD+a.SeasonalQ > 0 {
		name += "_" + strconv.Itoa(a.SeasonalP) + strconv.Itoa(a.SeasonalD) + strconv.Itoa(a.SeasonalQ)
	}
	if a.Regressors.Width() > 0 {
		name += "_x"
	}
	return name
}

func (a Arima) Description() string {
	d := "ARIMA(" + strconv.Itoa(a.P) + "," + strconv.Itoa(a.D) + "," + strconv.Itoa(a.Q) + ")"
	if a.SeasonalP+a.SeasonalD+a.SeasonalQ > 0 {
		d += "(" + strconv.Itoa(a.SeasonalP) + "," + strconv.Itoa(a.SeasonalD) + "," + strconv.Itoa(a.SeasonalQ) + ")"
	}
	d += " by exact maximum likelihood"
	if a.Regressors.Width() > 0 {
		d += ", with regressors"
	}
	return d
}

func (a Arima) Fit(y Series) (Fitted, error) {
	fit, err := a.Estimate(y)
	if err != nil {
		return nil, err
	}
	return fit, nil
}

// covers reports an error when the regressors do not reach the horizon.
func (a Arima) covers(y Series, h int) error {
	if !a.Regressors.Covers(y.Index(y.Len()) + h) {
		return ErrRegressors
	}
	return nil
}
