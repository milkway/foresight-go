package foresight

import (
	"errors"
	"math"
	"runtime"
	"sort"
	"strings"
	"sync"
)

// Candidate is a model taking part in a backtest, under a name of the
// caller's choice.
type Candidate struct {
	Name        string
	Description string
	Model       Model
}

// NewCandidate returns a candidate with the model's own name and
// description.
func NewCandidate(m Model) Candidate {
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
	// Window trains on the last Window observations only; zero uses
	// everything before the origin.
	Window int
	// Combine also evaluates the simple average of the best Combine models
	// (default 2; 1 disables it).
	Combine int
	// Levels is the coverage of the intervals, such as 0.8 for the 10%–90%
	// quantiles (default 0.80 and 0.95).
	Levels []float64
	// Metric ranks the candidates (default RankByMAPE).
	Metric Metric
	// Sequential fits the origins one at a time instead of on all cores. The
	// result is the same either way.
	Sequential bool
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
	MAE  float64
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

// Interval is an interval around a forecast.
type Interval struct {
	Level, Lower, Upper float64
}

// Point is a forecast with its intervals.
type Point struct {
	// Horizon is the number of periods ahead (for a cumulative forecast, how
	// many were added up).
	Horizon   int
	Mean      float64
	Intervals []Interval
}

func newPoint(horizon int, mean float64, bands []Band) Point {
	p := Point{Horizon: horizon, Mean: mean, Intervals: make([]Interval, len(bands))}
	for i, b := range bands {
		p.Intervals[i] = Interval{b.Level, mean * (1 + b.Lower), mean * (1 + b.Upper)}
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
	if k < 1 || k > len(c.Forecast) {
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
	Horizon     int
	Metric      Metric
}

// Best returns the chosen candidate.
func (r *Report) Best() *CandidateReport { return &r.Candidates[r.Chosen] }

// Candidate looks a candidate up by name.
func (r *Report) Candidate(name string) (*CandidateReport, bool) {
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
	// ErrNoCandidate: no candidate could be fitted at every origin.
	ErrNoCandidate = errors.New("foresight: no candidate could be fitted at every origin")
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

// mapIndices returns f(0), …, f(n − 1), computed on all cores unless
// sequential.
func mapIndices[T any](n int, sequential bool, f func(int) T) []T {
	out := make([]T, n)
	workers := min(runtime.NumCPU(), n)
	if sequential || workers <= 1 {
		for i := range out {
			out[i] = f(i)
		}
		return out
	}
	var wg sync.WaitGroup
	next := make(chan int)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				out[i] = f(i)
			}
		}()
	}
	for i := range n {
		next <- i
	}
	close(next)
	wg.Wait()
	return out
}

func (b Backtest) train(y Series, origin int) Series {
	if b.Window > 0 {
		return y.Slice(max(origin-b.Window, 0), origin)
	}
	return y.Head(origin)
}

func (b Backtest) stats(y Series, first int, trajectories [][]float64) []HorizonStats {
	v, n := y.Values(), y.Len()
	scales := make([]float64, len(trajectories))
	scaled := make([]bool, len(trajectories))
	for k := range trajectories {
		scales[k], scaled[k] = MASEScale(b.train(y, first+k).Values(), y.Period())
	}
	out := make([]HorizonStats, b.Horizon)
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
	}
	return out
}

func (b Backtest) score(horizons []HorizonStats) float64 {
	var values []float64
	for _, s := range horizons {
		v := s.MAPE
		switch b.Metric {
		case RankByMAE:
			v = s.MAE
		case RankByRMSE:
			v = s.RMSE
		case RankByMASE:
			v = s.MASE
		}
		if !math.IsNaN(v) {
			values = append(values, v)
		}
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
// It returns [ErrFewOrigins] when the series is too short for at least one
// pair at the longest horizon, and [ErrNoCandidate] when no candidate could
// be fitted at every origin.
func (b Backtest) Run(y Series, candidates []Candidate) (*Report, error) {
	b = b.filled()
	n := y.Len()
	origins := min(b.Origins, max(n-b.MinTrain, 0))
	if origins < b.Horizon {
		return nil, ErrFewOrigins
	}
	first := n - origins

	// 1. forecasts of every model at every origin
	var entries []entry
	for i, c := range candidates {
		forecasts := mapIndices(origins, b.Sequential, func(k int) []float64 {
			p, err := Forecast(c.Model, b.train(y, first+k), b.Horizon)
			if err != nil || len(p) != b.Horizon {
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
		if complete {
			entries = append(entries, entry{[]int{i}, forecasts})
		}
	}
	if len(entries) == 0 {
		return nil, ErrNoCandidate
	}
	horizons := make([][]HorizonStats, len(entries))
	scores := make([]float64, len(entries))
	for i, e := range entries {
		horizons[i] = b.stats(y, first, e.trajectories)
		scores[i] = b.score(horizons[i])
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
		stats := b.stats(y, first, trajectories)
		entries = append(entries, entry{components, trajectories})
		horizons = append(horizons, stats)
		scores = append(scores, b.score(stats))
	}

	// 3. choice by the average error over the horizons
	chosen := 0
	for i := range scores {
		if scores[i] < scores[chosen] {
			chosen = i
		}
	}

	// 4. forecast from the whole series, with each candidate's own bands
	finals := map[int][]float64{}
	params := map[int][]Param{}
	for _, e := range entries[:singles] {
		i := e.components[0]
		fit, err := candidates[i].Model.Fit(y)
		if err != nil {
			return nil, err
		}
		finals[i], params[i] = fit.Forecast(b.Horizon), fit.Params()
	}
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
	}
}
