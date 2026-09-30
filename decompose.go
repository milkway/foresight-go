package foresight

import (
	"math"
	"slices"
)

// Decomposition is a series split into trend, one seasonal pattern per
// period and remainder: the three add up to the series.
type Decomposition struct {
	// Periods are the seasonal periods taken out, in increasing order.
	Periods []int
	// Trend is the smooth part of the series.
	Trend []float64
	// Seasonal has one seasonal component per period, in the order of
	// Periods.
	Seasonal [][]float64
	// Remainder is what trend and seasonality leave unexplained.
	Remainder []float64
}

func sampleVariance(v []float64) float64 {
	n := float64(len(v))
	m := sum(v) / n
	s := 0.0
	for _, x := range v {
		s += (x - m) * (x - m)
	}
	return s / (n - 1)
}

// SeasonallyAdjusted returns the series without its seasonal patterns.
func (d Decomposition) SeasonallyAdjusted() []float64 {
	out := make([]float64, len(d.Trend))
	for t := range out {
		out[t] = d.Trend[t] + d.Remainder[t]
	}
	return out
}

func strength(part, remainder []float64) float64 {
	both := make([]float64, len(part))
	for t := range both {
		both[t] = part[t] + remainder[t]
	}
	total := sampleVariance(both)
	if total <= 0 {
		return 0
	}
	return min(max(1-sampleVariance(remainder)/total, 0), 1)
}

// TrendStrength returns the strength of the trend in [0, 1]:
// 1 − Var(remainder) / Var(trend + remainder) (Wang, Smith & Hyndman, 2006).
func (d Decomposition) TrendStrength() float64 {
	return strength(d.Trend, d.Remainder)
}

// SeasonalStrength returns the strength in [0, 1] of the seasonal component
// of index i. It reports false when there is no such component.
func (d Decomposition) SeasonalStrength(i int) (float64, bool) {
	if i < 0 || i >= len(d.Seasonal) {
		return 0, false
	}
	return strength(d.Seasonal[i], d.Remainder), true
}

// Stl is STL, the seasonal-trend decomposition by LOESS (Cleveland,
// Cleveland, McRae & Terpenning, 1990), for one seasonal period.
//
// The defaults are those of stl in R: trend window
// 1.5·period / (1 − 1.5/seasonal window), rounded up to the next odd number,
// low-pass window the next odd number after the period, local constant
// fitting for the seasonal pattern and local linear for the rest, and LOESS
// evaluated at every tenth of each window and interpolated in between.
//
// As in R, the series has to be longer than two full cycles.
type Stl struct {
	// Period is the seasonal period, at least 2.
	Period int
	// SeasonalWindow is the LOESS window over the cycles, an odd number,
	// usually 7 or more: the smaller, the faster the pattern may change. An
	// even number is taken as the next odd one, and the least is 3. Zero
	// keeps the same pattern in every cycle.
	SeasonalWindow int
	// TrendWindow and LowPassWindow are the LOESS windows of the trend and
	// of the low-pass filter; zero stands for the defaults.
	TrendWindow, LowPassWindow int
	// SeasonalLinear fits the seasonal pattern by local lines instead of
	// local constants; FlatTrend and FlatLowPass fit the trend and the
	// low-pass filter by local constants instead of local lines.
	SeasonalLinear, FlatTrend, FlatLowPass bool
	// Robust down-weights outliers: 15 robustness rounds of one pass each
	// instead of a single round of two passes.
	//
	// Compared with stl(robust = TRUE) in R, the result is the same to 12
	// digits for up to 11 rounds; from the 12th on the two drift apart in
	// the third digit, R's values no longer settling from one round to the
	// next.
	Robust bool
	// Inner and Outer are the passes of the inner loop and the robustness
	// rounds of the outer loop; zero stands for the defaults.
	Inner, Outer int
}

func nextOdd(x int) int {
	if x%2 == 0 {
		return x + 1
	}
	return x
}

type stlSetup struct {
	np, ns, nt, nl                         int
	seasonalDegree, trendDegree, lowDegree int
	seasonalJump, trendJump, lowPassJump   int
	inner                                  int
}

// Decompose decomposes the values. It returns an error for a period under 2,
// a series that is not longer than two full cycles (as in R's stl) or values
// that are not finite.
func (s Stl) Decompose(y []float64) (Decomposition, error) {
	n, np := len(y), s.Period
	if np < 2 {
		return Decomposition{}, ErrNotSeasonal
	}
	if n <= 2*np {
		return Decomposition{}, ErrTooShort
	}
	for _, v := range y {
		if !finite(v) {
			return Decomposition{}, ErrNotFinite
		}
	}
	periodic := s.SeasonalWindow <= 0
	span, seasonalDegree := s.SeasonalWindow, count(s.SeasonalLinear)
	if periodic {
		span, seasonalDegree = 10*n+1, 0
	}
	trend := s.TrendWindow
	if trend <= 0 {
		trend = nextOdd(int(math.Ceil(1.5 * float64(np) / (1 - 1.5/float64(span)))))
	}
	lowPass := s.LowPassWindow
	if lowPass <= 0 {
		lowPass = nextOdd(np)
	}
	jump := func(window int) int { return (window + 9) / 10 }
	setup := stlSetup{
		np: np, ns: nextOdd(max(span, 3)), nt: nextOdd(max(trend, 3)), nl: nextOdd(max(lowPass, 3)),
		seasonalDegree: seasonalDegree, trendDegree: 1 - count(s.FlatTrend), lowDegree: 1 - count(s.FlatLowPass),
		seasonalJump: jump(span), trendJump: jump(trend), lowPassJump: jump(lowPass),
		inner: s.Inner,
	}
	outer := s.Outer
	if setup.inner <= 0 {
		setup.inner = 2 - count(s.Robust)
	}
	if outer <= 0 {
		outer = 15 * count(s.Robust)
	}

	season := make([]float64, n)
	trendLine := make([]float64, n)
	var weights []float64
	for round := 0; round <= outer; round++ {
		setup.pass(y, weights, season, trendLine)
		if round == outer {
			break
		}
		fit := make([]float64, n)
		for t := range fit {
			fit[t] = trendLine[t] + season[t]
		}
		weights = robustnessWeights(y, fit)
	}
	if periodic {
		// the same pattern in every cycle: the mean of each position
		total := make([]float64, np)
		counts := make([]int, np)
		for t, v := range season {
			total[t%np] += v
			counts[t%np]++
		}
		for t := range season {
			season[t] = total[t%np] / float64(counts[t%np])
		}
	}
	remainder := make([]float64, n)
	for t := range remainder {
		remainder[t] = y[t] - season[t] - trendLine[t]
	}
	return Decomposition{
		Periods: []int{np}, Trend: trendLine, Seasonal: [][]float64{season}, Remainder: remainder,
	}, nil
}

// pass is the inner loop: detrend, smooth each cycle-subseries, take the
// low-frequency part out of the result, deseasonalise, smooth the trend.
func (s stlSetup) pass(y, weights, season, trend []float64) {
	n := len(y)
	for range s.inner {
		detrended := make([]float64, n)
		for t := range detrended {
			detrended[t] = y[t] - trend[t]
		}
		cycles := s.cycleSubseries(detrended, weights)
		filtered := movingAverage(movingAverage(movingAverage(cycles, s.np), s.np), 3)
		low := loess(filtered, s.nl, s.lowDegree, s.lowPassJump, nil)
		adjusted := make([]float64, n)
		for t := range adjusted {
			season[t] = cycles[s.np+t] - low[t]
			adjusted[t] = y[t] - season[t]
		}
		copy(trend, loess(adjusted, s.nt, s.trendDegree, s.trendJump, weights))
	}
}

// cycleSubseries smooths each cycle-subseries (all the Januaries, all the
// Februaries…) by LOESS and extends it by one cycle at each end: n + 2·period
// values.
func (s stlSetup) cycleSubseries(y, weights []float64) []float64 {
	n, np := len(y), s.np
	out := make([]float64, n+2*np)
	for j := range np {
		k := (n-j-1)/np + 1
		sub := make([]float64, k)
		var w []float64
		if weights != nil {
			w = make([]float64, k)
		}
		for i := range sub {
			sub[i] = y[i*np+j]
			if w != nil {
				w[i] = weights[i*np+j]
			}
		}
		smooth := loess(sub, s.ns, s.seasonalDegree, s.seasonalJump, w)
		before, ok := localFit(sub, s.ns, s.seasonalDegree, 0, 1, min(s.ns, k), w)
		if !ok {
			before = smooth[0]
		}
		after, ok := localFit(sub, s.ns, s.seasonalDegree, float64(k+1), max(k+1-s.ns, 1), k, w)
		if !ok {
			after = smooth[k-1]
		}
		out[j] = before
		for i, v := range smooth {
			out[(i+1)*np+j] = v
		}
		out[(k+1)*np+j] = after
	}
	return out
}

func movingAverage(x []float64, length int) []float64 {
	total := sum(x[:length])
	out := make([]float64, 0, len(x)-length+1)
	out = append(out, total/float64(length))
	for t := length; t < len(x); t++ {
		total += x[t] - x[t-length]
		out = append(out, total/float64(length))
	}
	return out
}

// robustnessWeights are bisquare weights from the size of the residuals: 1
// for the small ones, 0 beyond six times the median.
func robustnessWeights(y, fit []float64) []float64 {
	n := len(y)
	r := make([]float64, n)
	for i := range r {
		r[i] = math.Abs(y[i] - fit[i])
	}
	sorted := slices.Clone(r)
	slices.Sort(sorted)
	middle := n / 2
	// on data the fit reproduces exactly the residuals are rounding noise,
	// and telling them apart by size would discard points at random: the
	// yardstick never goes below what rounding leaves
	size := 0.0
	for _, v := range y {
		size += math.Abs(v)
	}
	size /= float64(n)
	cmad := max(3*(sorted[middle]+sorted[n-middle-1]), 1e-10*size)
	c9, c1 := 0.999*cmad, 0.001*cmad
	for i, v := range r {
		switch {
		case v <= c1:
			r[i] = 1
		case v <= c9:
			u := 1 - (v/cmad)*(v/cmad)
			r[i] = u * u
		default:
			r[i] = 0
		}
	}
	return r
}

func cube(x float64) float64 { return x * x * x }

// localFit is the local fit at the position at (the first value is at 1)
// from the values at positions left to right, with tricube weights. It
// reports false when every weight is zero.
func localFit(y []float64, window, degree int, at float64, left, right int, robustness []float64) (float64, bool) {
	n := len(y)
	span := float64(n) - 1
	h := math.Max(at-float64(left), float64(right)-at)
	if window > n {
		h += float64((window - n) / 2)
	}
	h9, h1 := 0.999*h, 0.001*h
	w := make([]float64, right-left+1)
	total := 0.0
	for j := left; j <= right; j++ {
		r := math.Abs(float64(j) - at)
		if r <= h9 {
			weight := 1.0
			if r > h1 {
				weight = cube(1 - cube(r/h))
			}
			if robustness != nil {
				weight *= robustness[j-1]
			}
			w[j-left] = weight
			total += weight
		}
	}
	if total <= 0 {
		return 0, false
	}
	for i := range w {
		w[i] /= total
	}
	if h > 0 && degree > 0 {
		centre, spread := 0.0, 0.0
		for j := left; j <= right; j++ {
			centre += w[j-left] * float64(j)
		}
		for j := left; j <= right; j++ {
			d := float64(j) - centre
			spread += w[j-left] * d * d
		}
		if math.Sqrt(spread) > 0.001*span {
			b := (at - centre) / spread
			for j := left; j <= right; j++ {
				w[j-left] *= b*(float64(j)-centre) + 1
			}
		}
	}
	out := 0.0
	for j := left; j <= right; j++ {
		out += w[j-left] * y[j-1]
	}
	return out, true
}

// loess smooths the whole series, evaluated every jump positions and
// interpolated in between.
func loess(y []float64, window, degree, jump int, robustness []float64) []float64 {
	n := len(y)
	if n < 2 {
		return slices.Clone(y)
	}
	jump = min(max(jump, 1), n-1)
	out := make([]float64, n)
	left, right := 1, min(n, window)
	half := (window + 1) / 2
	at := func(i int) float64 {
		if v, ok := localFit(y, window, degree, float64(i), left, right, robustness); ok {
			return v
		}
		return y[i-1]
	}
	switch {
	case window >= n:
		left, right = 1, n
		for i := 1; i <= n; i += jump {
			out[i-1] = at(i)
		}
	case jump == 1:
		for i := 1; i <= n; i++ {
			if i > half && right != n {
				left++
				right++
			}
			out[i-1] = at(i)
		}
	default:
		for i := 1; i <= n; i += jump {
			switch {
			case i < half:
				left, right = 1, window
			case i > n-half:
				left, right = n-window+1, n
			default:
				left, right = i-half+1, window+i-half
			}
			out[i-1] = at(i)
		}
	}
	if jump != 1 {
		for i := 1; i <= n-jump; i += jump {
			slope := (out[i+jump-1] - out[i-1]) / float64(jump)
			for j := i + 1; j < i+jump; j++ {
				out[j-1] = out[i-1] + slope*float64(j-i)
			}
		}
		if last := ((n-1)/jump)*jump + 1; last != n {
			out[n-1] = at(n)
			if last != n-1 {
				slope := (out[n-1] - out[last-1]) / float64(n-last)
				for j := last + 1; j < n; j++ {
					out[j-1] = out[last-1] + slope*float64(j-last)
				}
			}
		}
	}
	return out
}

// Mstl is STL applied in turn to each of several seasonal periods (Bandara,
// Hyndman & Bergmeir, 2021).
//
// The periods are taken from the shortest to the longest; each pattern is
// estimated on the series without the others, twice over. Periods that do
// not fit more than twice in the series are left out. The trend comes from
// the last fit.
type Mstl struct {
	// Periods are the seasonal periods; those under 2 are left out.
	Periods []int
	// Windows are the seasonal windows, one per period in increasing order
	// of period; the last one is repeated if there are more periods than
	// windows. The default is 11, 15, 19… for the first, second, third…
	// period.
	Windows []int
	// Iterations are the rounds over the periods (default 2).
	Iterations int
	// Robust down-weights outliers in every fit.
	Robust bool
}

// Decompose decomposes the values. It returns an error when the series is
// not longer than two cycles of any of the periods, or the values are not
// finite.
func (m Mstl) Decompose(y []float64) (Decomposition, error) {
	n := len(y)
	for _, v := range y {
		if !finite(v) {
			return Decomposition{}, ErrNotFinite
		}
	}
	var periods []int
	for _, p := range m.Periods {
		if p >= 2 && 2*p < n {
			periods = append(periods, p)
		}
	}
	slices.Sort(periods)
	periods = slices.Compact(periods)
	if len(periods) == 0 {
		return Decomposition{}, ErrTooShort
	}
	windows := m.Windows
	if len(windows) == 0 {
		windows = []int{11, 15, 19, 23, 27, 31}
	}
	seasonal := make([][]float64, len(periods))
	for i := range seasonal {
		seasonal[i] = make([]float64, n)
	}
	rest := slices.Clone(y)
	var trend []float64
	rounds := m.Iterations
	if rounds <= 0 {
		rounds = 2
	}
	for range rounds {
		for i, period := range periods {
			for t := range rest {
				rest[t] += seasonal[i][t]
			}
			fit, err := Stl{Period: period, SeasonalWindow: windows[min(i, len(windows)-1)], Robust: m.Robust}.Decompose(rest)
			if err != nil {
				return Decomposition{}, err
			}
			copy(seasonal[i], fit.Seasonal[0])
			for t := range rest {
				rest[t] -= seasonal[i][t]
			}
			trend = fit.Trend
		}
	}
	remainder := make([]float64, n)
	for t := range remainder {
		remainder[t] = rest[t] - trend[t]
	}
	return Decomposition{Periods: periods, Trend: trend, Seasonal: seasonal, Remainder: remainder}, nil
}

// Decomposed runs a model on the seasonally adjusted series and adds the
// seasonal patterns of the last cycle to its forecasts.
//
// The decomposition is [Mstl], so the series may have several seasonal
// periods; by default the period is the one of the series. The model inside
// sees a series without seasonality, at the same position in the original
// data, so what goes by position (regressors, events, a deflator) stays
// aligned. The series has to be longer than two cycles of a period for it
// to be taken out. For patterns that grow with the level, wrap the whole in
// [Log].
type Decomposed struct {
	// Model is the model of the seasonally adjusted series.
	Model Model
	// Periods are the seasonal periods to take out, instead of the period of
	// the series.
	Periods []int
	// Robust down-weights outliers in the decomposition.
	Robust bool
}

type decomposedFit struct {
	inner Fitted
	// the last cycle of each seasonal pattern
	cycles [][]float64
}

func (f decomposedFit) Forecast(h int) []float64 {
	if h <= 0 {
		return nil
	}
	out := f.inner.Forecast(h)
	for k := range out {
		for _, c := range f.cycles {
			out[k] += c[k%len(c)]
		}
	}
	return out
}

func (f decomposedFit) Params() []Param {
	return append(f.inner.Params(), Param{"seasonal_periods", float64(len(f.cycles))})
}

// Name is the identifier of the model: "stl_" and the name of the model
// inside ("stl_invalid" without one).
func (d Decomposed) Name() string { return "stl_" + nameOf(d.Model) }

// Description is a one-line description of the model.
func (d Decomposed) Description() string {
	return descriptionOf(d.Model) + ", on the series seasonally adjusted by STL"
}

// Fit decomposes the series and fits the model inside on what is left. It
// returns [ErrConfig] without a model inside.
func (d Decomposed) Fit(y Series) (Fitted, error) {
	if d.Model == nil {
		return nil, ErrConfig
	}
	periods := d.Periods
	if periods == nil {
		periods = []int{y.Period()}
	}
	if slices.Max(append([]int{0}, periods...)) < 2 {
		// nothing to take out
		return d.Model.Fit(y)
	}
	parts, err := Mstl{Periods: periods, Robust: d.Robust}.Decompose(y.Values())
	if err != nil {
		return nil, err
	}
	inner, err := d.Model.Fit(y.flat(parts.SeasonallyAdjusted()))
	if err != nil {
		return nil, err
	}
	n := y.Len()
	cycles := make([][]float64, len(parts.Periods))
	for i, p := range parts.Periods {
		cycles[i] = slices.Clone(parts.Seasonal[i][n-p:])
	}
	return decomposedFit{inner, cycles}, nil
}
