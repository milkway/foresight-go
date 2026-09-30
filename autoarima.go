package foresight

import "slices"

// Criterion is the information criterion that ranks models.
type Criterion int

const (
	// ByAICc ranks by the AIC corrected for small samples (the default).
	ByAICc Criterion = iota
	// ByAIC ranks by the AIC.
	ByAIC
	// ByBIC ranks by the BIC.
	ByBIC
)

// AutoArima is seasonal ARIMA with orders chosen from the data (Hyndman &
// Khandakar, 2008).
//
// The differences come first: a seasonal difference when the seasonality is
// strong ([NSDiffs]), then as many ordinary differences as the KPSS test
// asks for ([NDiffs]). The remaining orders are found by a stepwise search:
// starting from four simple models, the best one is varied — each order up
// or down by one, alone and in pairs, with and without a constant — for as
// long as the information criterion improves. Models with roots too close to
// the unit circle are discarded.
//
// As a [Model], the orders are chosen again at every fit, so a backtest
// judges the whole procedure, not one lucky specification.
//
// The zero value has the usual limits; see [DefaultAutoArima].
type AutoArima struct {
	// Limits of the search. When all six are zero they stand for the
	// defaults: 5, 5, 2, 2, 2 and 1.
	MaxP, MaxQ, MaxSeasonalP, MaxSeasonalQ, MaxD, MaxSeasonalD int
	// FixDifferences uses D and SeasonalD instead of testing.
	FixDifferences bool
	D, SeasonalD   int
	Criterion      Criterion
	// MaxModels is the most models the search will fit (default 94).
	MaxModels int
	// Regressors are external variables of a regression with ARIMA errors;
	// the differences are then decided on what the regression leaves
	// unexplained.
	Regressors Regressors
}

// DefaultAutoArima returns the defaults spelled out.
func DefaultAutoArima() AutoArima {
	return AutoArima{MaxP: 5, MaxQ: 5, MaxSeasonalP: 2, MaxSeasonalQ: 2, MaxD: 2, MaxSeasonalD: 1, MaxModels: 94}
}

func (a AutoArima) filled() AutoArima {
	d := DefaultAutoArima()
	if a.MaxP == 0 && a.MaxQ == 0 && a.MaxSeasonalP == 0 && a.MaxSeasonalQ == 0 && a.MaxD == 0 && a.MaxSeasonalD == 0 {
		a.MaxP, a.MaxQ, a.MaxSeasonalP, a.MaxSeasonalQ = d.MaxP, d.MaxQ, d.MaxSeasonalP, d.MaxSeasonalQ
		a.MaxD, a.MaxSeasonalD = d.MaxD, d.MaxSeasonalD
	}
	if a.MaxModels <= 0 {
		a.MaxModels = d.MaxModels
	}
	return a
}

// unexplained returns the residuals of the least squares regression of the
// series on a constant and the regressors.
func (a AutoArima) unexplained(y Series) ([]float64, error) {
	v := y.Values()
	if a.Regressors.Width() == 0 {
		return v, nil
	}
	if !a.Regressors.Covers(y.Index(len(v))) {
		return nil, ErrRegressors
	}
	columns := [][]float64{repeat(1, len(v))}
	for _, c := range a.Regressors.columns {
		column := make([]float64, len(v))
		for t := range column {
			column[t] = c[y.Index(t)]
		}
		columns = append(columns, column)
	}
	gram := make([][]float64, len(columns))
	moment := make([]float64, len(columns))
	for i, ci := range columns {
		gram[i] = make([]float64, len(columns))
		for j, cj := range columns {
			gram[i][j] = dot(ci, cj)
		}
		moment[i] = dot(ci, v)
	}
	beta, ok := solve(gram, moment)
	if !ok {
		return nil, ErrRegressors
	}
	out := make([]float64, len(v))
	for t := range out {
		fit := 0.0
		for j, c := range columns {
			fit += c[t] * beta[j]
		}
		out[t] = v[t] - fit
	}
	return out, nil
}

func (a AutoArima) score(f *ArimaFit) float64 {
	switch a.Criterion {
	case ByAIC:
		return f.AIC
	case ByBIC:
		return f.BIC
	}
	return f.AICc
}

// (p, q, P, Q, constant)
type arimaKey struct {
	p, q, sp, sq int
	constant     bool
}

// Select chooses the orders and returns the estimated model.
func (a AutoArima) Select(y Series) (*ArimaFit, error) {
	a = a.filled()
	if y.Len() == 0 {
		return nil, ErrTooShort
	}
	if !y.IsFinite() {
		return nil, ErrNotFinite
	}
	m := y.Period()
	v, err := a.unexplained(y)
	if err != nil {
		return nil, err
	}
	seasonal := m > 1
	constantSeries := true
	for _, x := range v {
		if x != v[0] {
			constantSeries = false
			break
		}
	}
	sd, d := 0, 0
	if seasonal && !constantSeries {
		if a.FixDifferences {
			sd = a.SeasonalD
		} else {
			sd = min(NSDiffs(v, m), a.MaxSeasonalD)
		}
	}
	if !constantSeries {
		if a.FixDifferences {
			d = a.D
		} else {
			w := v
			for range sd {
				w = Difference(w, m)
			}
			d = NDiffs(w, a.MaxD)
		}
	}
	withConstant := d+sd <= 1

	var tried []arimaKey
	var best *ArimaFit
	var bestKey arimaKey
	bestScore := 0.0
	model := func(k arimaKey) Arima {
		c := WithoutConstant
		if k.constant {
			c = WithConstant
		}
		return Arima{P: k.p, D: d, Q: k.q, SeasonalP: k.sp, SeasonalD: sd, SeasonalQ: k.sq,
			Constant: c, Regressors: a.Regressors}
	}
	consider := func(k arimaKey) bool {
		if k.p > a.MaxP || k.q > a.MaxQ || k.sp > a.MaxSeasonalP || k.sq > a.MaxSeasonalQ ||
			(k.constant && !withConstant) || (!seasonal && (k.sp > 0 || k.sq > 0)) ||
			slices.Contains(tried, k) || len(tried) >= a.MaxModels {
			return false
		}
		tried = append(tried, k)
		fit, err := model(k).estimateFrom(y, best)
		if err != nil {
			return false
		}
		score := a.score(fit)
		if !finite(score) || !fit.IsWellBehaved(1.01) {
			return false
		}
		if best == nil || score < bestScore {
			best, bestKey, bestScore = fit, k, score
			return true
		}
		return false
	}

	s := 0
	if seasonal {
		s = 1
	}
	for _, k := range []arimaKey{{2, 2, s, s, withConstant}, {0, 0, 0, 0, withConstant}, {1, 0, s, 0, withConstant}, {0, 1, 0, s, withConstant}} {
		consider(k)
	}
	if withConstant {
		consider(arimaKey{})
	}
	for {
		if best == nil {
			return nil, ErrNoFit
		}
		k := bestKey
		var neighbours []arimaKey
		for _, step := range []int{-1, 1} {
			// each order alone, then the pairs; none below zero
			for _, n := range []arimaKey{
				{k.p + step, k.q, k.sp, k.sq, k.constant},
				{k.p, k.q + step, k.sp, k.sq, k.constant},
				{k.p, k.q, k.sp + step, k.sq, k.constant},
				{k.p, k.q, k.sp, k.sq + step, k.constant},
				{k.p + step, k.q + step, k.sp, k.sq, k.constant},
				{k.p, k.q, k.sp + step, k.sq + step, k.constant},
			} {
				if n.p >= 0 && n.q >= 0 && n.sp >= 0 && n.sq >= 0 {
					neighbours = append(neighbours, n)
				}
			}
		}
		neighbours = append(neighbours, arimaKey{k.p, k.q, k.sp, k.sq, !k.constant})
		improved := false
		for _, n := range neighbours {
			if consider(n) {
				improved = true
				break
			}
		}
		if !improved {
			break
		}
	}
	// the winner once more from the usual starting points, in case the
	// search settled on a lesser peak of its likelihood
	if again, err := model(bestKey).Estimate(y); err == nil && a.score(again) < bestScore && again.IsWellBehaved(1.01) {
		return again, nil
	}
	return best, nil
}

func (AutoArima) Name() string { return "auto_arima" }
func (AutoArima) Description() string {
	return "ARIMA with differences chosen by tests and orders by stepwise search on the information criterion"
}

func (a AutoArima) Fit(y Series) (Fitted, error) {
	fit, err := a.Select(y)
	if err != nil {
		return nil, err
	}
	return fit, nil
}

func (a AutoArima) covers(y Series, h int) error {
	if !a.Regressors.Covers(y.Index(y.Len()) + h) {
		return ErrRegressors
	}
	return nil
}
