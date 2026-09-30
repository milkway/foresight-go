package foresight

import (
	"errors"
	"math"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
)

// Candidate is a model taking part in a backtest, under a name of the
// caller's choice.
type Candidate struct {
	// Name identifies the candidate in the report.
	Name string
	// Description is a line about the candidate, for display.
	Description string
	// Model is what is fitted. A candidate without one takes no part.
	Model Model
}

// NewCandidate returns a candidate with the model's own name and
// description. A nil model gives the zero Candidate, which takes no part in
// a backtest.
func NewCandidate(m Model) Candidate {
	if m == nil {
		return Candidate{}
	}
	return Candidate{Name: m.Name(), Description: m.Description(), Model: m}
}

// Named returns the candidate under another name and description.
func (c Candidate) Named(name, description string) Candidate {
	c.Name, c.Description = name, description
	return c
}

// Metric is the error measure averaged over the horizons to rank the
// candidates.
type Metric int

const (
	// RankByMAPE ranks by mean absolute percentage error (the default).
	RankByMAPE Metric = iota
	// RankByMAE ranks by mean absolute error.
	RankByMAE
	// RankByRMSE ranks by root mean squared error.
	RankByRMSE
	// RankByMASE ranks by mean absolute scaled error.
	RankByMASE
)

// Backtest is the configuration of a rolling-origin evaluation.
//
//  1. Rolling origin: for each of the last Origins periods, every candidate
//     is fitted ONLY on the data before the origin and forecasts 1 to Horizon
//     periods ahead; errors are measured against what happened. Horizon h has
//     Origins − h + 1 pairs.
//  2. Selection: the candidate with the lowest error averaged over the
//     horizons, the simple average of the best few models included.
//  3. Empirical intervals: no normality assumed. For each horizon, the
//     quantiles of the relative error r = actual/forecast − 1 seen in the
//     backtest; the interval is forecast × (1 + q).
//  4. Cumulative intervals: the same for the ratio of sums over the first k
//     periods of each origin, because adding up the limits of k intervals
//     overstates the uncertainty of a total.
//
// The zero value of each field stands for its default; [DefaultBacktest]
// returns them spelled out.
type Backtest struct {
	// Origins is how many of the last periods are forecast origins
	// (default 36).
	Origins int
	// Horizon is the longest horizon forecast and evaluated (default 12).
	Horizon int
	// MinTrain is the number of training observations at the first origin
	// (default 48). Shorter series get fewer origins.
	MinTrain int
	// Window trains on the last Window observations only, at every origin
	// and for the final forecast; zero uses everything before the origin.
	Window int
	// Combine also evaluates the simple average of the best Combine models
	// (default 2; 1 disables it).
	Combine int
	// Levels is the coverage of the intervals, such as 0.8 for the 10%–90%
	// quantiles, each one above 0 and below 1 (default 0.80 and 0.95).
	Levels []float64
	// Metric ranks the candidates (default RankByMAPE).
	Metric Metric
	// Sequential fits the origins one at a time, on the calling goroutine,
	// instead of on several cores (see [SetMaxThreads]); the ensembles among
	// the candidates then start no goroutines either. The result is the same
	// either way.
	Sequential bool
}

// maxThreads is the most goroutines a computation may use; 0 stands for
// every core.
var maxThreads atomic.Int64

// SetMaxThreads limits the goroutines used at a time by each backtest and
// each ensemble, in the whole process. Zero, the default, stands for every
// core Go may use (GOMAXPROCS); a negative number is read as zero.
//
// The results do not depend on the number of goroutines.
func SetMaxThreads(n int) {
	maxThreads.Store(int64(max(n, 0)))
}

// MaxThreads returns the most goroutines a backtest or an ensemble will use:
// what was set with [SetMaxThreads], or the number of cores Go may use.
func MaxThreads() int {
	if limit := int(maxThreads.Load()); limit > 0 {
		return limit
	}
	return max(runtime.GOMAXPROCS(0), 1)
}

// DefaultBacktest returns the defaults, which suit monthly data: 36 origins,
// 12 periods ahead, at least 48 observations to train, the average of the two
// best models, intervals of 80% and 95%.
func DefaultBacktest() Backtest {
	return Backtest{
		Origins: 36, Horizon: 12, MinTrain: 48, Combine: 2,
		Levels: []float64{0.80, 0.95},
	}
}

func (b Backtest) filled() Backtest {
	d := DefaultBacktest()
	if b.Origins <= 0 {
		b.Origins = d.Origins
	}
	if b.Horizon <= 0 {
		b.Horizon = d.Horizon
	}
	if b.MinTrain <= 0 {
		b.MinTrain = d.MinTrain
	}
	if b.Combine <= 0 {
		b.Combine = d.Combine
	}
	if b.Levels == nil {
		b.Levels = d.Levels
	}
	return b
}

// Band holds the quantiles of a relative error r (actual/forecast − 1) for
// one coverage level: the interval is forecast × (1 + Lower) to
// forecast × (1 + Upper).
type Band struct {
	Level, Lower, Upper float64
}

// HorizonStats is what the backtest measured at one horizon. Measures that
// could not be computed are NaN.
type HorizonStats struct {
	// N is the number of forecast/actual pairs evaluated.
	N int
	// MAPE is the mean absolute percentage error, in percent.
	MAPE float64
	// Bias is the mean of (forecast − actual)/actual, in percent; positive
	// when the forecasts ran high.
	Bias float64
	// MAE is the mean absolute error.
	MAE float64
	// RMSE is the root mean squared error.
	RMSE float64
	// MASE is the mean absolute error scaled by the in-sample seasonal naive
	// error of each training set.
	MASE float64
	// Bands are the quantiles of the relative error at this horizon, one per
	// level.
	Bands []Band
	// Cumulative are the quantiles of the relative error of the SUM of the
	// first h periods.
	Cumulative []Band
}

// Interval is an interval around a forecast, with its coverage: Lower is
// never above Upper.
type Interval struct {
	Level, Lower, Upper float64
}

// Point is a forecast with its intervals.
type Point struct {
	// Horizon is the number of periods ahead (for a cumulative forecast, how
	// many were added up).
	Horizon int
	// Mean is the point forecast.
	Mean float64
	// Intervals has one interval per level of the backtest.
	Intervals []Interval
}

func newPoint(horizon int, mean float64, bands []Band) Point {
	p := Point{Horizon: horizon, Mean: mean, Intervals: make([]Interval, len(bands))}
	for i, b := range bands {
		// a negative forecast turns the bounds around
		a, z := mean*(1+b.Lower), mean*(1+b.Upper)
		p.Intervals[i] = Interval{b.Level, min(a, z), max(a, z)}
	}
	return p
}

// Interval returns the interval with the given coverage, if it was asked
// for.
func (p Point) Interval(level float64) (Interval, bool) {
	for _, i := range p.Intervals {
		if math.Abs(i.Level-level) < 1e-9 {
			return i, true
		}
	}
	return Interval{}, false
}

// CandidateReport is the backtest and the forecast of one candidate (a model
// or an average of models).
type CandidateReport struct {
	// Name and Description are those of the candidate; an average is named
	// after its components.
	Name        string
	Description string
	// Components are the names of the models involved: one, or several for
	// an average.
	Components []string
	// Horizons has the measures by horizon; index 0 is horizon 1.
	Horizons []HorizonStats
	// Score is the ranking metric averaged over the horizons; +Inf when it
	// could not be computed.
	Score float64
	// Params are the parameters fitted on the whole series (none for an
	// average).
	Params []Param
	// Trajectories are the backtest forecasts, [origin][h − 1].
	Trajectories [][]float64
	// Forecast is the forecast from the whole series, with the intervals of
	// the backtest.
	Forecast []Point
}

// Cumulative returns the forecast of the SUM of the first k periods, with
// intervals from the backtest of sums. It reports false when k is under 1 or
// beyond the horizon.
func (c CandidateReport) Cumulative(k int) (Point, bool) {
	if k < 1 || k > len(c.Forecast) || k > len(c.Horizons) {
		return Point{}, false
	}
	total := 0.0
	for _, p := range c.Forecast[:k] {
		total += p.Mean
	}
	return newPoint(k, total, c.Horizons[k-1].Cumulative), true
}

// Report is the result of a backtest.
type Report struct {
	// Candidates are the models in the order given, then the average of the
	// best ones.
	Candidates []CandidateReport
	// Chosen is the index of the chosen candidate.
	Chosen int
	// Origins is the number of origins actually used.
	Origins int
	// FirstOrigin is the position in the series of the first period
	// forecast in the backtest.
	FirstOrigin int
	// Horizon is the longest horizon forecast and evaluated.
	Horizon int
	// Metric is the measure that ranked the candidates.
	Metric Metric
}

// Best returns the chosen candidate. For a report that did not come from
// [Backtest.Run] (no candidates, or Chosen out of range) it returns an empty
// CandidateReport.
func (r *Report) Best() *CandidateReport {
	if r == nil || r.Chosen < 0 || r.Chosen >= len(r.Candidates) {
		return &CandidateReport{}
	}
	return &r.Candidates[r.Chosen]
}

// Candidate looks a candidate up by name.
func (r *Report) Candidate(name string) (*CandidateReport, bool) {
	if r == nil {
		return nil, false
	}
	for i := range r.Candidates {
		if r.Candidates[i].Name == name {
			return &r.Candidates[i], true
		}
	}
	return nil, false
}

// Errors of a backtest.
var (
	// ErrFewOrigins: the series is too short for at least one pair at the
	// longest horizon.
	ErrFewOrigins = errors.New("foresight: series too short for the backtest")
	// ErrNoCandidate: no candidate gave finite forecasts at every origin and
	// from the whole series.
	ErrNoCandidate = errors.New("foresight: no candidate could be fitted at every origin")
	// ErrLevels: a level of the intervals is not above 0 and below 1.
	ErrLevels = errors.New("foresight: levels must be above 0 and below 1")
)

func bands(errs, levels []float64) []Band {
	var out []Band
	for _, level := range levels {
		tail := (1 - level) / 2
		lo, ok1 := Quantile(errs, tail)
		hi, ok2 := Quantile(errs, 1-tail)
		if ok1 && ok2 {
			out = append(out, Band{level, lo, hi})
		}
	}
	return out
}

// threads returns how many goroutines a computation of n independent parts
// may use: at most [MaxThreads], and only the calling one when alone, which
// is how a computation started from inside another (an ensemble in a worker
// of a backtest) is told not to multiply the goroutines.
func threads(n int, alone bool) int {
	if alone {
		return 1
	}
	return max(min(MaxThreads(), n), 1)
}

// mapIndices returns f(0), …, f(n − 1), computed on the given number of
// goroutines, the calling one included.
func mapIndices[T any](n, workers int, f func(int) T) []T {
	out := make([]T, n)
	if workers <= 1 {
		for i := range out {
			out[i] = f(i)
		}
		return out
	}
	var next atomic.Int64
	work := func() {
		for {
			i := int(next.Add(1)) - 1
			if i >= n {
				return
			}
			out[i] = f(i)
		}
	}
	var wg sync.WaitGroup
	for range min(workers, n) - 1 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			work()
		}()
	}
	// the calling goroutine works too
	work()
	wg.Wait()
	return out
}

func (b Backtest) train(y Series, origin int) Series {
	if b.Window > 0 {
		return y.Slice(max(origin-b.Window, 0), origin)
	}
	return y.Head(origin)
}

// stats returns the measures by horizon and, for each horizon, how many
// pairs the ranking metric was computed from.
func (b Backtest) stats(y Series, first int, trajectories [][]float64) ([]HorizonStats, []int) {
	v, n := y.Values(), y.Len()
	scales := make([]float64, len(trajectories))
	scaled := make([]bool, len(trajectories))
	for k := range trajectories {
		scales[k], scaled[k] = MASEScale(b.train(y, first+k).Values(), y.Period())
	}
	out := make([]HorizonStats, b.Horizon)
	pairs := make([]int, b.Horizon)
	for h := 1; h <= b.Horizon; h++ {
		var absPct, pct, abs, sq, mase, rel, cum []float64
		for k, forecast := range trajectories {
			origin := first + k // trained on y[:origin]
			target := origin + h - 1
			if target >= n {
				continue
			}
			actual, predicted := v[target], forecast[h-1]
			abs = append(abs, math.Abs(predicted-actual))
			sq = append(sq, (predicted-actual)*(predicted-actual))
			if scaled[k] {
				mase = append(mase, math.Abs(predicted-actual)/scales[k])
			}
			if actual != 0 {
				absPct = append(absPct, math.Abs(predicted-actual)/math.Abs(actual))
				pct = append(pct, (predicted-actual)/actual)
			}
			if predicted != 0 {
				rel = append(rel, actual/predicted-1)
			}
			if sumPredicted := sum(forecast[:h]); sumPredicted != 0 {
				cum = append(cum, sum(v[origin:target+1])/sumPredicted-1)
			}
		}
		out[h-1] = HorizonStats{
			N:          len(abs),
			MAPE:       mean(absPct) * 100,
			Bias:       mean(pct) * 100,
			MAE:        mean(abs),
			RMSE:       math.Sqrt(mean(sq)),
			MASE:       mean(mase),
			Bands:      bands(rel, b.Levels),
			Cumulative: bands(cum, b.Levels),
		}
		switch b.Metric {
		case RankByMAE, RankByRMSE:
			pairs[h-1] = len(abs)
		case RankByMASE:
			pairs[h-1] = len(mase)
		default:
			pairs[h-1] = len(absPct)
		}
	}
	return out, pairs
}

// score is the ranking metric averaged over the horizons that have pairs to
// compute it from; +Inf when there is none or when it is not finite at one
// of them, so that what cannot be measured never ranks first.
func (b Backtest) score(horizons []HorizonStats, pairs []int) float64 {
	var values []float64
	for h, s := range horizons {
		if pairs[h] == 0 {
			continue
		}
		v := s.MAPE
		switch b.Metric {
		case RankByMAE:
			v = s.MAE
		case RankByRMSE:
			v = s.RMSE
		case RankByMASE:
			v = s.MASE
		}
		if !finite(v) {
			return math.Inf(1)
		}
		values = append(values, v)
	}
	if m := mean(values); finite(m) {
		return m
	}
	return math.Inf(1)
}

type entry struct {
	components   []int // indices into the candidates given
	trajectories [][]float64
}

// Run evaluates the candidates on y, chooses one and forecasts Horizon
// periods with every candidate that went through the whole backtest.
//
// A candidate takes part only if it gives Horizon finite forecasts at every
// origin and from the whole series; the others (those without a model, those
// that fail, those that panic) are left out of the report.
//
// It returns [ErrFewOrigins] when the series is too short for at least one
// pair at the longest horizon, [ErrLevels] when a level is not above 0 and
// below 1, and [ErrNoCandidate] when no candidate is left. There is no other
// error.
func (b Backtest) Run(y Series, candidates []Candidate) (*Report, error) {
	b = b.filled()
	n := y.Len()
	origins := min(b.Origins, max(n-b.MinTrain, 0))
	if origins < b.Horizon {
		return nil, ErrFewOrigins
	}
	for _, level := range b.Levels {
		if !(level > 0 && level < 1) {
			return nil, ErrLevels
		}
	}
	first := n - origins
	usable := func(p []float64) bool {
		if len(p) != b.Horizon {
			return false
		}
		for _, v := range p {
			if !finite(v) {
				return false
			}
		}
		return true
	}
	workers := threads(origins, b.Sequential)
	// what a worker fits stays on its goroutine; with Sequential, everything
	// stays on this one
	atOrigin, whole := y, y
	if workers > 1 || b.Sequential {
		atOrigin = y.onOneGoroutine()
	}
	if b.Sequential {
		whole = atOrigin
	}

	// 1. forecasts of every model at every origin, and from the whole series
	//    (the last Window observations of it) with the parameters of that fit
	var entries []entry
	finals := map[int][]float64{}
	params := map[int][]Param{}
	for i, c := range candidates {
		if c.Model == nil {
			continue
		}
		forecasts := mapIndices(origins, workers, func(k int) []float64 {
			p, err := tryForecast(c.Model, b.train(atOrigin, first+k), b.Horizon)
			if err != nil || !usable(p) {
				return nil
			}
			return p
		})
		complete := true
		for _, p := range forecasts {
			if p == nil {
				complete = false
				break
			}
		}
		if !complete {
			continue
		}
		forecast, fitted, err := tryFit(c.Model, b.train(whole, n), b.Horizon)
		if err != nil || !usable(forecast) {
			continue
		}
		finals[i], params[i] = forecast, fitted
		entries = append(entries, entry{[]int{i}, forecasts})
	}
	if len(entries) == 0 {
		return nil, ErrNoCandidate
	}
	horizons := make([][]HorizonStats, len(entries))
	scores := make([]float64, len(entries))
	for i, e := range entries {
		var pairs []int
		horizons[i], pairs = b.stats(y, first, e.trajectories)
		scores[i] = b.score(horizons[i], pairs)
	}

	// 2. simple average of the best models
	singles := len(entries)
	order := make([]int, singles)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, c int) bool { return scores[order[a]] < scores[order[c]] })
	if b.Combine >= 2 && singles >= b.Combine {
		best := order[:b.Combine]
		trajectories := make([][]float64, origins)
		for o := range trajectories {
			trajectories[o] = make([]float64, b.Horizon)
			for h := range trajectories[o] {
				total := 0.0
				for _, i := range best {
					total += entries[i].trajectories[o][h]
				}
				trajectories[o][h] = total / float64(len(best))
			}
		}
		components := make([]int, len(best))
		for j, i := range best {
			components[j] = entries[i].components[0]
		}
		stats, pairs := b.stats(y, first, trajectories)
		entries = append(entries, entry{components, trajectories})
		horizons = append(horizons, stats)
		scores = append(scores, b.score(stats, pairs))
	}

	// 3. choice by the average error over the horizons
	chosen := 0
	for i := range scores {
		if scores[i] < scores[chosen] {
			chosen = i
		}
	}

	// 4. forecast from the whole series, with each candidate's own bands
	reports := make([]CandidateReport, len(entries))
	for j, e := range entries {
		names := make([]string, len(e.components))
		for k, i := range e.components {
			names[k] = candidates[i].Name
		}
		forecast := make([]Point, b.Horizon)
		for h := range forecast {
			total := 0.0
			for _, i := range e.components {
				total += finals[i][h]
			}
			forecast[h] = newPoint(h+1, total/float64(len(e.components)), horizons[j][h].Bands)
		}
		r := CandidateReport{
			Components: names, Horizons: horizons[j], Score: scores[j],
			Trajectories: e.trajectories, Forecast: forecast,
		}
		if len(e.components) == 1 {
			i := e.components[0]
			r.Name, r.Description, r.Params = names[0], candidates[i].Description, params[i]
		} else {
			r.Name = "mean(" + strings.Join(names, "+") + ")"
			r.Description = "Simple average of " + strings.Join(names, " and ")
		}
		reports[j] = r
	}
	return &Report{
		Candidates: reports, Chosen: chosen, Origins: origins,
		FirstOrigin: first, Horizon: b.Horizon, Metric: b.Metric,
	}, nil
}

// Defaults returns a reasonable set of candidates for a backtest: the
// benchmarks and every model that needs no external data and fits in a
// moment.
func Defaults() []Candidate {
	return []Candidate{
		NewCandidate(Naive{}),
		NewCandidate(Drift{}),
		NewCandidate(SeasonalNaive{}),
		NewCandidate(SeasonalNaive{Growth: true}),
		NewCandidate(Theta{}),
		NewCandidate(HoltWinters{}),
		NewCandidate(LogLinear{}),
		NewCandidate(Airline()),
		NewCandidate(Log(Airline())),
		NewCandidate(Prophet{}),
		NewCandidate(Log(Prophet{})),
	}
}

// Thorough returns [Defaults] plus two ensembles of them, exponential
// smoothing chosen automatically, alone and after an STL decomposition, and
// ARIMA with automatic orders, on the original and on the log scale. The
// choices are made again at every origin of the backtest, which takes seconds
// rather than milliseconds.
func Thorough() []Candidate {
	return append(Defaults(),
		NewCandidate(Ensemble{Members: Defaults()}),
		NewCandidate(Ensemble{Members: Defaults(), Weighting: Stacked}),
		NewCandidate(AutoEts{}),
		NewCandidate(Decomposed{Model: AutoEts{}}),
		NewCandidate(Log(Decomposed{Model: AutoEts{}})),
		NewCandidate(AutoArima{}),
		NewCandidate(Log(AutoArima{})),
	)
}
