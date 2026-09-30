package foresight

// Inputs at the edges: what used to give wrong answers, no answer at all or a
// panic. The first tests are those of tests/edge_cases.rs in the Rust crate;
// the rest are about what only the Go edition could get wrong.

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sync/atomic"
	"testing"
	"time"
)

// edgeSeasonal is a seasonal series with growth and a small wobble.
func edgeSeasonal(n int) []float64 {
	out := make([]float64, n)
	for t := range out {
		wobble := 1 + 0.02*(float64(t*37%11)/10-0.5)
		out[t] = 100 * math.Pow(1.005, float64(t)) * pattern[t%12] * wobble
	}
	return out
}

// lengthModel forecasts the number of observations it was fitted on.
type lengthModel struct{}

func (lengthModel) Name() string        { return "length" }
func (lengthModel) Description() string { return "the number of observations" }
func (lengthModel) Fit(y Series) (Fitted, error) {
	return constant(float64(y.Len())), nil
}

// nanModel forecasts what is not a number; it would have the lowest error of
// all if the horizons it cannot be measured at were skipped.
type nanModel struct{}

func (nanModel) Name() string               { return "nan" }
func (nanModel) Description() string        { return "not a number" }
func (nanModel) Fit(Series) (Fitted, error) { return constant(math.NaN()), nil }

// panicModel panics when fitted.
type panicModel struct{}

func (panicModel) Name() string               { return "panic" }
func (panicModel) Description() string        { return "panics when fitted" }
func (panicModel) Fit(Series) (Fitted, error) { panic("a model gone wrong") }

// latePanic is fitted without trouble at every origin and panics on the
// whole series, whose length it is told.
type latePanic struct{ whole int }

type panicFit struct{}

func (panicFit) Forecast(int) []float64 { panic("a fit gone wrong") }
func (panicFit) Params() []Param        { return nil }

func (latePanic) Name() string        { return "late_panic" }
func (latePanic) Description() string { return "panics when asked for the final forecast" }
func (l latePanic) Fit(y Series) (Fitted, error) {
	if y.Len() == l.whole {
		return panicFit{}, nil
	}
	return Naive{}.Fit(y)
}

// probe is the naive model, watching how it is used: how many fits run at
// the same time and whether the series came marked for one goroutine.
type probe struct {
	running, most, unmarked *atomic.Int64
}

func newProbe() probe {
	return probe{new(atomic.Int64), new(atomic.Int64), new(atomic.Int64)}
}

func (probe) Name() string        { return "probe" }
func (probe) Description() string { return "naive, watched" }
func (p probe) Fit(y Series) (Fitted, error) {
	now := p.running.Add(1)
	for {
		most := p.most.Load()
		if now <= most || p.most.CompareAndSwap(most, now) {
			break
		}
	}
	if !y.alone {
		p.unmarked.Add(1)
	}
	time.Sleep(100 * time.Microsecond)
	p.running.Add(-1)
	return Naive{}.Fit(y)
}

// printed is a report as text, which is equal for equal numbers, those that
// are not numbers included.
func printed(r *Report) string { return fmt.Sprintf("%+v", *r) }

// quiet fails the test if f panics.
func quiet(t *testing.T, what string, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("%s panicked: %v", what, r)
		}
	}()
	f()
}

func TestEdgeCandidateWithoutFinalForecastIsLeftOut(t *testing.T) {
	// the last observation is in no training set: multiplicative models go
	// through every origin and then cannot be fitted on the whole series
	y := edgeSeasonal(120)
	y[119] = 0
	candidates := []Candidate{NewCandidate(Naive{}), NewCandidate(HoltWinters{}), NewCandidate(Log(Drift{}))}
	r, err := DefaultBacktest().Run(Monthly(y, 0), candidates)
	if err != nil {
		t.Fatalf("the naive forecast is still there: %v", err)
	}
	if _, ok := r.Candidate("naive"); !ok {
		t.Error("naive left out")
	}
	for _, name := range []string{"holt_winters", "log_drift"} {
		if _, ok := r.Candidate(name); ok {
			t.Errorf("%s has no final forecast and is in the report", name)
		}
	}
	if _, err := DefaultBacktest().Run(Monthly(y, 0), Defaults()); err != nil {
		t.Errorf("default candidates: %v", err)
	}
}

func TestEdgeShortRegressorsLeaveTheCandidateOut(t *testing.T) {
	y := edgeSeasonal(120)
	h := 12
	x := make([]float64, 120+h-1)
	for i := range x {
		x[i] = math.Sin(float64(i) * 0.7)
	}
	short := Arima{P: 1, Regressors: Regressors{}.With("x", x)}
	r, err := DefaultBacktest().Run(Monthly(y, 0), []Candidate{NewCandidate(short), NewCandidate(Naive{})})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Candidates) != 1 {
		t.Errorf("%d candidates", len(r.Candidates))
	}
	for _, c := range r.Candidates {
		for _, p := range c.Forecast {
			if !finite(p.Mean) {
				t.Errorf("%s: forecast %v", c.Name, p.Mean)
			}
		}
	}
	// on its own and inside another model, an error rather than NaN
	series := Monthly(y, 0)
	for _, m := range []Model{short, Log(short), Decomposed{Model: short}, Ensemble{Members: []Candidate{NewCandidate(short)}, Weighting: EqualWeights}} {
		if p, err := Forecast(m, series, h); err == nil {
			t.Errorf("%s: forecast %v", m.Name(), p)
		}
	}
	if _, err := Forecast(Decomposed{Model: short}, series, h); !errors.Is(err, ErrForecast) {
		t.Errorf("inside a decomposition: %v", err)
	}
	if _, err := Forecast(short, series, h-1); err != nil {
		t.Errorf("one period less is covered: %v", err)
	}
}

func TestEdgeIntervalsOfNegativeForecastsAreInOrder(t *testing.T) {
	y := edgeSeasonal(120)
	for i := range y {
		y[i] = -y[i]
	}
	r, err := DefaultBacktest().Run(Monthly(y, 0), []Candidate{NewCandidate(Mean{}), NewCandidate(Naive{})})
	if err != nil {
		t.Fatal(err)
	}
	// the naive forecast is about right, so it is inside its own interval
	naive, _ := r.Candidate("naive")
	p := naive.Forecast[0]
	if i, ok := p.Interval(0.8); !ok || !(i.Lower < p.Mean && p.Mean < i.Upper) {
		t.Errorf("%+v around %v", i, p.Mean)
	}
	for _, c := range r.Candidates {
		for _, p := range c.Forecast {
			if p.Mean >= 0 {
				t.Errorf("%s: forecast %v", c.Name, p.Mean)
			}
			for _, i := range p.Intervals {
				if !(i.Lower <= i.Upper) {
					t.Errorf("%s: %+v", c.Name, i)
				}
			}
		}
		total, _ := c.Cumulative(6)
		if i, ok := total.Interval(0.8); !ok || !(i.Lower <= i.Upper) {
			t.Errorf("%s, cumulative: %+v", c.Name, i)
		}
	}
}

func TestEdgeLevelsHaveToBeBetweenZeroAndOne(t *testing.T) {
	y := Monthly(edgeSeasonal(120), 0)
	for _, bad := range []float64{0, 1, 1.5, -0.5, math.NaN()} {
		_, err := Backtest{Levels: []float64{0.8, bad}}.Run(y, []Candidate{NewCandidate(Naive{})})
		if !errors.Is(err, ErrLevels) {
			t.Errorf("level %v: %v", bad, err)
		}
	}
	// the series being too short is said first
	if _, err := (Backtest{Levels: []float64{2}}).Run(y.Head(50), classic()); !errors.Is(err, ErrFewOrigins) {
		t.Errorf("short series: %v", err)
	}
}

func TestEdgeWindowAlsoHoldsForTheFinalForecast(t *testing.T) {
	r, err := Backtest{Window: 60}.Run(Monthly(edgeSeasonal(120), 0), []Candidate{NewCandidate(lengthModel{})})
	if err != nil {
		t.Fatal(err)
	}
	c := r.Candidates[0]
	for _, trajectory := range c.Trajectories {
		if trajectory[0] != 60 {
			t.Fatalf("an origin trained on %v observations", trajectory[0])
		}
	}
	if c.Forecast[0].Mean != 60 {
		t.Errorf("the final forecast trained on %v observations", c.Forecast[0].Mean)
	}
}

func TestEdgeSeasonalPatternNeedsTwoCycles(t *testing.T) {
	// eight months between 50 and 60: nothing to say about the seasons
	y := Monthly([]float64{52, 57, 51, 59, 55, 53, 58, 54}, 0)
	for _, m := range []Model{Prophet{}, Log(Prophet{})} {
		f, err := Forecast(m, y, 12)
		if err != nil {
			t.Fatalf("%s: %v", m.Name(), err)
		}
		for _, v := range f {
			if v < 30 || v >= 90 {
				t.Errorf("%s: %v", m.Name(), f)
				break
			}
		}
	}
	if (Prophet{}).filled().harmonics(12, 23) != 0 || (Prophet{}).filled().harmonics(12, 24) != 6 {
		t.Error("seasonal terms from two full cycles on")
	}
}

func TestEdgeAutoArimaKeepsAnExactFit(t *testing.T) {
	constant := repeat(5, 60)
	f, err := Forecast(AutoArima{}, Monthly(constant, 0), 3)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range f {
		if math.Abs(v-5) >= 1e-9 {
			t.Errorf("constant series: %v", f)
		}
	}
	line := make([]float64, 60)
	for i := range line {
		line[i] = 10 + 2*float64(i+1)
	}
	f, err = Forecast(AutoArima{}, NonSeasonal(line), 3)
	if err != nil {
		t.Fatal(err)
	}
	for k, v := range f {
		if math.Abs(v-(132+2*float64(k))) >= 1e-6 {
			t.Errorf("straight line: %v", f)
		}
	}
	// the criteria of an exact fit are finite
	fit, err := Arima{D: 1, Constant: WithConstant}.Estimate(NonSeasonal(line))
	if err != nil {
		t.Fatal(err)
	}
	if !finite(fit.LogLikelihood) || !finite(fit.AIC) || !finite(fit.AICc) || !finite(fit.BIC) {
		t.Errorf("criteria %v %v %v %v", fit.LogLikelihood, fit.AIC, fit.AICc, fit.BIC)
	}
	// a fit is as exact when the differences leave rounding noise (0.1 has
	// no exact form in binary), and among exact fits the smallest model wins
	for i := range line {
		line[i] = 10.3 + 0.1*float64(i+1)
	}
	fit, err = Arima{D: 1, Constant: WithConstant}.Estimate(NonSeasonal(line))
	if err != nil || fit.LogLikelihood != 0.5e300 || !finite(fit.AICc) {
		t.Errorf("rounding noise taken for something to explain: %v, %v", fit.LogLikelihood, err)
	}
	fit, err = AutoArima{FixDifferences: true, D: 1}.Select(NonSeasonal(line))
	if err != nil {
		t.Fatal(err)
	}
	if p, _, q := fit.Order(); p != 0 || q != 0 || !fit.HasConstant {
		t.Errorf("ARIMA(%d,1,%d), constant %v, for a straight line", p, q, fit.HasConstant)
	}
	for k, v := range fit.Forecast(3) {
		if math.Abs(v-(10.3+0.1*float64(61+k))) >= 1e-9 {
			t.Errorf("straight line, h = %d: %v", k+1, v)
		}
	}
}

func TestEdgeDecomposedKeepsItsPlaceInTime(t *testing.T) {
	// a step of −30 from position 80 on, in a series with a seasonal pattern
	y := make([]float64, 120)
	for i := range y {
		y[i] = 200 + 0.5*float64(i) + 10*(pattern[i%12]-1)*10
		if i >= 80 {
			y[i] -= 30
		}
	}
	model := Decomposed{Model: Prophet{NoChangepoints: true, Events: []Event{{Name: "law", Step: true, StepFrom: 80}}}}
	whole := must(t, model, Monthly(y, 0), 6)
	// from observation 40 on, the step is still at position 80 of the data
	part := must(t, model, Monthly(y, 0).Slice(40, 120), 6)
	for k := range whole {
		if math.Abs(whole[k]-part[k]) >= 5 {
			t.Fatalf("%v against %v", whole, part)
		}
	}
}

func TestEdgeExactDataHasNoOutliers(t *testing.T) {
	line := make([]float64, 60)
	for i := range line {
		line[i] = 10 + 0.5*float64(i)
	}
	for _, period := range []int{1, 12} {
		for name, v := range map[string][]float64{"constant": repeat(3, 40), "line": line} {
			if found, ok := Outliers(v, period); !ok || len(found) != 0 {
				t.Errorf("%s, period %d: %v %v", name, period, found, ok)
			}
		}
	}
	// and one that is there is still found
	spiked := slices.Clone(line)
	spiked[30] += 50
	found, ok := Outliers(spiked, 1)
	if !ok || len(found) != 1 || found[0].Index != 30 || math.Abs(found[0].Replacement-line[30]) >= 1 {
		t.Errorf("spike: %+v %v", found, ok)
	}
	// in a robust decomposition, rounding noise does not decide which points
	// count: only the one that is off loses its weight
	fit := slices.Clone(line)
	for i := range fit {
		fit[i] += 1e-13 * float64(i%7-3)
	}
	for i, w := range robustnessWeights(spiked, fit) {
		if (i == 30) != (w == 0) || (i != 30 && w != 1) {
			t.Fatalf("weight %v at %d", w, i)
		}
	}
}

func TestEdgeSmallThings(t *testing.T) {
	// growth of a cycle that adds up to a negative number means nothing
	y := edgeSeasonal(36)
	for i := 24; i < 36; i++ {
		y[i] = -y[i]
	}
	if _, err := (SeasonalNaive{Growth: true}).Fit(Monthly(y, 0)); !errors.Is(err, ErrNotPositive) {
		t.Errorf("growth from a negative cycle: %v", err)
	}
	// a smoothing that the method does not use is not checked
	demand := NonSeasonal([]float64{0, 0, 3, 0, 2, 0, 0, 4})
	if _, err := (Croston{Beta: 2}).Fit(demand); err != nil {
		t.Errorf("croston with a beta it does not use: %v", err)
	}
	if _, err := (Croston{Variant: TSB, Beta: 2}).Fit(demand); err == nil {
		t.Error("TSB with a beta of 2")
	}
	// harmonics beyond half the period repeat the earlier ones
	if Fourier(12, 12, 24).Width() != 11 || Fourier(12, 6, 24).Width() != 11 || Fourier(7, 5, 24).Width() != 6 {
		t.Error("harmonics beyond half the period")
	}
	// a forecast that is not finite is no forecast
	up := make([]float64, 60)
	for i := range up {
		up[i] = float64(i + 1)
	}
	if p, err := Forecast(WithBoxCox(Drift{}, -1), NonSeasonal(up), 5); !errors.Is(err, ErrForecast) {
		t.Errorf("a scale that cannot be brought back: %v %v", p, err)
	}
}

func TestEdgeConstantSeriesDoesNotSendTbatsSearching(t *testing.T) {
	started := time.Now()
	fit, err := Tbats{}.Select(Monthly(repeat(7, 60), 0))
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range fit.Forecast(3) {
		if math.Abs(v-7) >= 1e-6 {
			t.Errorf("forecast %v", fit.Forecast(3))
		}
	}
	// milliseconds; a search that takes rounding noise for something to
	// explain needs a hundred times as long
	if spent := time.Since(started); spent > 5*time.Second {
		t.Errorf("took %v", spent)
	}
}

// From here on, what only the Go edition could get wrong.

func TestEdgeCandidateThatIsNotANumberDoesNotWin(t *testing.T) {
	y := Monthly(noisySeasonalGrowth(112, 0.05), 0)
	r, err := DefaultBacktest().Run(y, append(classic(), NewCandidate(nanModel{})))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := r.Candidate("nan"); ok || !finite(r.Best().Score) || len(r.Candidates) != 4 {
		t.Errorf("chosen %s with score %v among %d", r.Best().Name, r.Best().Score, len(r.Candidates))
	}
	if _, err := DefaultBacktest().Run(y, []Candidate{NewCandidate(nanModel{})}); !errors.Is(err, ErrNoCandidate) {
		t.Errorf("alone: %v", err)
	}
	// and a measure that is not a number, where there were pairs to compute
	// it from, is the worst score rather than a horizon left out
	b := DefaultBacktest()
	horizons := []HorizonStats{{N: 3, MAPE: 2}, {N: 3, MAPE: math.NaN()}, {N: 0, MAPE: math.NaN()}}
	if s := b.score(horizons, []int{3, 3, 0}); !math.IsInf(s, 1) {
		t.Errorf("score %v with a measure that is not a number", s)
	}
	if s := b.score(horizons, []int{3, 0, 0}); s != 2 {
		t.Errorf("score %v when the horizon has no pairs", s)
	}
	if s := b.score(horizons[2:], []int{0}); !math.IsInf(s, 1) {
		t.Errorf("score %v without any pair", s)
	}
}

func TestEdgeQuantile(t *testing.T) {
	v := []float64{3, 1, 2}
	quiet(t, "Quantile(NaN)", func() {
		if q, ok := Quantile(v, math.NaN()); ok || q != 0 {
			t.Errorf("quantile NaN: %v %v", q, ok)
		}
	})
	if q, ok := Quantile(v, -3); !ok || q != 1 {
		t.Errorf("p under 0: %v", q)
	}
	if q, ok := Quantile(v, math.Inf(1)); !ok || q != 3 {
		t.Errorf("p over 1: %v", q)
	}
	// values that are not numbers go last, as in the Rust crate
	holes := []float64{math.NaN(), 4, 1, 3, 2}
	if q, _ := Quantile(holes, 0); q != 1 {
		t.Errorf("lowest of %v: %v", holes, q)
	}
	if q, _ := Quantile(holes, 0.5); q != 3 {
		t.Errorf("median of %v: %v", holes, q)
	}
	if q, ok := Quantile(holes, 1); !ok || !math.IsNaN(q) {
		t.Errorf("highest of %v: %v", holes, q)
	}
	if totalOrder(math.Copysign(0, -1), 0) >= 0 || totalOrder(math.Inf(1), math.NaN()) >= 0 ||
		totalOrder(-math.NaN(), math.Inf(-1)) >= 0 || totalOrder(1, 1) != 0 {
		t.Error("total order")
	}
}

// edgeModels are the models of the package, one for each kind of fit, with
// cheap settings where there is a choice.
func edgeModels() []Model {
	quick := Defaults()[:4]
	return []Model{
		Mean{}, Naive{}, Drift{}, SeasonalNaive{}, SeasonalNaive{Growth: true}, Theta{}, HoltWinters{},
		LogLinear{}, Airline(), AutoArima{MaxP: 1, MaxQ: 1, MaxSeasonalP: -1, MaxSeasonalQ: -1, MaxModels: 3},
		Ets{}, Ets{Trend: DampedTrend, Season: AdditiveSeason}, Prophet{}, Croston{}, Croston{Variant: TSB},
		Tbats{Harmonics: []int{1}, BoxCox: No, Trend: No, ArmaErrors: No},
		Log(Drift{}), WithGuerrero(Naive{}), Decomposed{Model: Drift{}},
		Ensemble{Members: quick}, Ensemble{Members: quick, Weighting: Median},
	}
}

func TestEdgeNegativeHorizon(t *testing.T) {
	y := Monthly(noisySeasonalGrowth(48, 0.03), 0)
	for _, m := range edgeModels() {
		if _, err := Forecast(m, y, -1); !errors.Is(err, ErrHorizon) {
			t.Errorf("%s: forecast of -1 periods: %v", m.Name(), err)
		}
		fit, err := m.Fit(y)
		if err != nil {
			t.Errorf("%s: %v", m.Name(), err)
			continue
		}
		quiet(t, m.Name(), func() {
			for _, h := range []int{-1, 0, math.MinInt} {
				if p := fit.Forecast(h); len(p) != 0 {
					t.Errorf("%s: %d forecasts for h = %d", m.Name(), len(p), h)
				}
			}
			if p := fit.Forecast(2); len(p) != 2 {
				t.Errorf("%s: %d forecasts for h = 2", m.Name(), len(p))
			}
		})
	}
	if p, err := Forecast(Naive{}, y, 0); err != nil || len(p) != 0 {
		t.Errorf("no periods: %v %v", p, err)
	}
	arima, _ := Airline().Estimate(y)
	if v := arima.ForecastVariance(-1); len(v) != 0 {
		t.Errorf("variance of -1 periods: %v", v)
	}
}

func TestEdgeZeroSeries(t *testing.T) {
	var y Series
	quiet(t, "the zero series", func() {
		if y.Period() != 1 || y.Season(0) != 0 || y.Season(7) != 0 || y.WithPhase(3).Season(2) != 0 || y.Len() != 0 {
			t.Error("the zero series is an empty series without seasonality")
		}
		if y.Head(3).Len() != 0 || y.Tail(3).Len() != 0 || y.Slice(-2, 5).Len() != 0 || !y.IsFinite() {
			t.Error("slices of the zero series")
		}
		if _, ok := Guerrero(y); ok {
			t.Error("guerrero on nothing")
		}
	})
	if _, err := (SeasonalNaive{}).Fit(y); !errors.Is(err, ErrTooShort) {
		t.Errorf("seasonal naive on nothing: %v", err)
	}
	for _, m := range append(edgeModels(), AutoArima{}, AutoEts{}, Tbats{}) {
		quiet(t, m.Name(), func() {
			if fit, err := m.Fit(y); err == nil {
				t.Errorf("%s: fitted on nothing: %v", m.Name(), fit.Forecast(1))
			}
			if _, err := Forecast(m, y, 3); err == nil {
				t.Errorf("%s: forecast from nothing", m.Name())
			}
		})
	}
	quiet(t, "a backtest of nothing", func() {
		if _, err := DefaultBacktest().Run(y, Defaults()); !errors.Is(err, ErrFewOrigins) {
			t.Errorf("backtest of nothing: %v", err)
		}
	})
}

func TestEdgeAutoArimaLimitsAreFilledOneByOne(t *testing.T) {
	d := DefaultAutoArima()
	got := AutoArima{MaxP: 3}.filled()
	if got.MaxP != 3 || got.MaxQ != d.MaxQ || got.MaxSeasonalP != d.MaxSeasonalP || got.MaxSeasonalQ != d.MaxSeasonalQ ||
		got.MaxD != d.MaxD || got.MaxSeasonalD != d.MaxSeasonalD || got.MaxModels != d.MaxModels {
		t.Errorf("limits %+v", got)
	}
	got = AutoArima{MaxQ: -1, MaxD: -1, MaxSeasonalD: -7, MaxModels: -1}.filled()
	if got.MaxP != d.MaxP || got.MaxQ != 0 || got.MaxD != 0 || got.MaxSeasonalD != 0 || got.MaxModels != d.MaxModels {
		t.Errorf("limits %+v", got)
	}
	if got = (AutoArima{}).filled(); got.MaxP != 5 || got.MaxQ != 5 || got.MaxSeasonalP != 2 || got.MaxSeasonalQ != 2 ||
		got.MaxD != 2 || got.MaxSeasonalD != 1 || got.MaxModels != 94 {
		t.Errorf("the zero value has the defaults: %+v", got)
	}
	// a random walk with drift: one limit set, and it is still differenced
	walk := make([]float64, 120)
	for i := 1; i < len(walk); i++ {
		walk[i] = walk[i-1] + 1 + math.Sin(float64(i)*1.7) + 0.5*math.Cos(float64(i)*0.3)
	}
	fit, err := AutoArima{MaxP: 3}.Select(NonSeasonal(walk))
	if err != nil {
		t.Fatal(err)
	}
	if p, d, _ := fit.Order(); d != 1 || p > 3 {
		t.Errorf("orders (%d, %d, ·) with MaxP of 3", p, d)
	}
	// the same on a seasonal series: the seasonal difference is kept too
	seasonal, err := AutoArima{MaxP: 1, MaxModels: 8}.Select(Monthly(logs(noisySeasonalGrowth(120, 0.02)), 0))
	if err != nil {
		t.Fatal(err)
	}
	if _, sd, _ := seasonal.SeasonalOrder(); sd != 1 {
		t.Errorf("seasonal difference %d with MaxP of 1", sd)
	}
	// and a negative limit is none
	fit, err = AutoArima{MaxD: -1, MaxQ: -1}.Select(NonSeasonal(walk))
	if err != nil {
		t.Fatal(err)
	}
	if _, d, q := fit.Order(); d != 0 || q != 0 {
		t.Errorf("orders (·, %d, %d) with no difference and no moving average allowed", d, q)
	}
}

func TestEdgeInvalidConfigurations(t *testing.T) {
	y := Monthly(noisySeasonalGrowth(72, 0.03), 0)
	members := Defaults()[:3]
	invalid := map[string]Model{
		"arima, negative p":            Arima{P: -1},
		"arima, negative d":            Arima{D: -1},
		"arima, negative q":            Arima{Q: -2},
		"arima, negative seasonal p":   Arima{SeasonalP: -1},
		"arima, negative seasonal d":   Arima{SeasonalD: -1},
		"arima, negative seasonal q":   Arima{SeasonalQ: -1},
		"arima, unknown constant":      Arima{Constant: 7},
		"auto arima, negative d":       AutoArima{FixDifferences: true, D: -1},
		"auto arima, negative D":       AutoArima{FixDifferences: true, SeasonalD: -1},
		"tbats, negative p":            Tbats{P: -1},
		"tbats, negative q":            Tbats{FixOrders: true, Q: -2},
		"ets, unknown error":           Ets{Error: 5},
		"ets, unknown trend":           Ets{Trend: 9},
		"ets, unknown season":          Ets{Season: -1},
		"ensemble, unknown weighting":  Ensemble{Members: members, Weighting: 9},
		"ensemble, negative weighting": Ensemble{Members: members, Weighting: -1},
		"ensemble, member missing":     Ensemble{Members: []Candidate{NewCandidate(Naive{}), {Name: "nothing"}}},
		"croston, unknown variant":     Croston{Variant: 9},
		"croston, negative variant":    Croston{Variant: -1},
		"log of nothing":               Log(nil),
		"box-cox of nothing":           Transformed{Automatic: true},
		"decomposition of nothing":     Decomposed{},
		"log of a log of nothing":      Log(Log(nil)),
		"prophet, range not a number":  Prophet{ChangepointRange: math.NaN()},
	}
	for what, m := range invalid {
		quiet(t, what, func() {
			if m.Name() == "" || m.Description() == "" {
				t.Errorf("%s: no name or description", what)
			}
			if _, err := m.Fit(y); !errors.Is(err, ErrConfig) {
				t.Errorf("%s: fit: %v", what, err)
			}
			if _, err := Forecast(m, y, 3); !errors.Is(err, ErrConfig) {
				t.Errorf("%s: forecast: %v", what, err)
			}
		})
	}
	for name, m := range map[string]Model{
		"ets_invalid": Ets{Trend: 9}, "ensemble_invalid": Ensemble{Weighting: 9}, "croston_invalid": Croston{Variant: 9},
		"log_invalid": Log(nil), "boxcox_invalid": WithBoxCox(nil, 0.5), "stl_invalid": Decomposed{},
	} {
		if m.Name() != name {
			t.Errorf("%s is called %s", name, m.Name())
		}
	}
	if (Ets{Season: 4}).Code() != "invalid" {
		t.Error("code of an invalid model")
	}
	if _, err := Forecast(nil, y, 3); !errors.Is(err, ErrConfig) {
		t.Errorf("forecast of no model: %v", err)
	}
	if _, err := (Ensemble{}).Fit(y); !errors.Is(err, ErrNoCandidate) {
		t.Errorf("ensemble of nothing: %v", err)
	}
	// in a backtest, what cannot be fitted is left out
	quiet(t, "backtest", func() {
		candidates := []Candidate{NewCandidate(nil), {Name: "nothing"}, NewCandidate(Log(nil)), NewCandidate(Arima{P: -1}), NewCandidate(Naive{})}
		r, err := Backtest{Origins: 12, Horizon: 3, MinTrain: 24}.Run(y, candidates)
		if err != nil || len(r.Candidates) != 1 || r.Best().Name != "naive" {
			t.Errorf("report %+v, %v", r, err)
		}
		if _, err := DefaultBacktest().Run(y, candidates[:4]); !errors.Is(err, ErrNoCandidate) {
			t.Errorf("no candidate that can be fitted: %v", err)
		}
	})
}

func TestEdgeModelThatPanicsIsLeftOut(t *testing.T) {
	values := noisySeasonalGrowth(112, 0.05)
	y := Monthly(values, 0)
	candidates := append(classic(),
		NewCandidate(panicModel{}), NewCandidate(latePanic{len(values)}),
		NewCandidate(Log(panicModel{})), NewCandidate(Ensemble{Members: []Candidate{NewCandidate(panicModel{})}}),
		NewCandidate(Ensemble{Members: []Candidate{NewCandidate(panicModel{}), NewCandidate(Naive{})}}).Named("ensemble_of_two", ""),
	)
	want, err := DefaultBacktest().Run(y, classic())
	if err != nil {
		t.Fatal(err)
	}
	for _, sequential := range []bool{false, true} {
		r, err := Backtest{Sequential: sequential}.Run(y, candidates)
		if err != nil {
			t.Fatalf("sequential %v: %v", sequential, err)
		}
		for _, name := range []string{"panic", "late_panic", "log_panic", "ensemble_inverse_error"} {
			if _, ok := r.Candidate(name); ok {
				t.Errorf("sequential %v: %s is in the report", sequential, name)
			}
		}
		// an ensemble goes on without the member that panics
		if _, ok := r.Candidate("ensemble_of_two"); !ok {
			t.Errorf("sequential %v: the ensemble with a sound member was left out", sequential)
		}
		for _, c := range want.Candidates[:3] {
			got, ok := r.Candidate(c.Name)
			if !ok || got.Score != c.Score {
				t.Errorf("sequential %v: %s changed in bad company", sequential, c.Name)
			}
		}
	}
	if _, err := DefaultBacktest().Run(y, []Candidate{NewCandidate(panicModel{})}); !errors.Is(err, ErrNoCandidate) {
		t.Errorf("alone: %v", err)
	}
}

func TestEdgeThreads(t *testing.T) {
	defer SetMaxThreads(0)
	SetMaxThreads(0)
	if MaxThreads() < 1 {
		t.Errorf("%d threads", MaxThreads())
	}
	SetMaxThreads(3)
	if MaxThreads() != 3 || threads(8, false) != 3 || threads(2, false) != 2 || threads(8, true) != 1 || threads(0, false) != 1 {
		t.Error("the limit of threads")
	}
	SetMaxThreads(-5)
	if MaxThreads() < 1 {
		t.Error("a negative limit is no limit")
	}
	for _, workers := range []int{0, 1, 2, 7, 50} {
		squares := mapIndices(20, workers, func(i int) int { return i * i })
		for i, v := range squares {
			if v != i*i {
				t.Fatalf("%d workers: %v", workers, squares)
			}
		}
		if len(mapIndices(0, workers, func(i int) int { return i })) != 0 {
			t.Errorf("%d workers on nothing", workers)
		}
	}

	// the same report whatever the number of goroutines
	y := Monthly(noisySeasonalGrowth(112, 0.05), 0)
	candidates := append(Defaults()[:4],
		NewCandidate(LogLinear{}),
		NewCandidate(Ensemble{Members: Defaults()[:4]}),
		NewCandidate(Ensemble{Members: Defaults()[:4], Weighting: Stacked}),
		NewCandidate(Log(Ensemble{Members: Defaults()[:3], Weighting: Median})),
	)
	SetMaxThreads(0)
	want, err := DefaultBacktest().Run(y, candidates)
	if err != nil {
		t.Fatal(err)
	}
	if len(want.Candidates) != len(candidates)+1 {
		t.Fatalf("%d candidates in the report", len(want.Candidates))
	}
	for _, limit := range []int{1, 3} {
		SetMaxThreads(limit)
		got, err := DefaultBacktest().Run(y, candidates)
		if err != nil || printed(got) != printed(want) {
			t.Errorf("the report changes with %d threads (%v)", limit, err)
		}
	}
	SetMaxThreads(0)
	got, err := Backtest{Sequential: true}.Run(y, candidates)
	if err != nil || printed(got) != printed(want) {
		t.Errorf("the report of a sequential backtest is another (%v)", err)
	}
}

func TestEdgeGoroutinesDoNotMultiply(t *testing.T) {
	defer SetMaxThreads(0)
	y := Monthly(noisySeasonalGrowth(112, 0.05), 0)
	backtest := Backtest{Origins: 18, Horizon: 6}
	ensemble := func(p probe) []Candidate {
		inner := Ensemble{Members: []Candidate{NewCandidate(p), NewCandidate(Drift{})}, Origins: 6}
		return []Candidate{
			NewCandidate(p),
			NewCandidate(Ensemble{Members: []Candidate{NewCandidate(p), NewCandidate(Naive{})}, Origins: 6}),
			// and an ensemble inside a model inside an ensemble
			NewCandidate(Ensemble{Members: []Candidate{NewCandidate(Log(inner)), NewCandidate(Naive{})}, Weighting: Stacked, Origins: 6}),
		}
	}

	// alone, an ensemble uses the goroutines it may
	SetMaxThreads(3)
	p := newProbe()
	if _, err := (Ensemble{Members: []Candidate{NewCandidate(p)}}).Fit(y); err != nil {
		t.Fatal(err)
	}
	if most := p.most.Load(); most < 1 || most > 3 {
		t.Errorf("an ensemble ran %d fits at a time with a limit of 3", most)
	}

	// inside a parallel backtest it stays on the goroutine of its worker
	p = newProbe()
	if _, err := backtest.Run(y, ensemble(p)); err != nil {
		t.Fatal(err)
	}
	if most := p.most.Load(); most < 1 || most > 3 {
		t.Errorf("%d fits at a time with a limit of 3", most)
	}

	// Sequential holds for the ensembles among the candidates
	SetMaxThreads(0)
	p = newProbe()
	backtest.Sequential = true
	if _, err := backtest.Run(y, ensemble(p)); err != nil {
		t.Fatal(err)
	}
	if most, unmarked := p.most.Load(), p.unmarked.Load(); most != 1 || unmarked != 0 {
		t.Errorf("sequential backtest: %d fits at a time, %d of them free to start goroutines", most, unmarked)
	}

	// and so does a limit of one goroutine
	SetMaxThreads(1)
	backtest.Sequential = false
	p = newProbe()
	if _, err := backtest.Run(y, ensemble(p)); err != nil {
		t.Fatal(err)
	}
	if most := p.most.Load(); most != 1 {
		t.Errorf("limit of one: %d fits at a time", most)
	}

	// the mark follows the series through slices and other values
	marked := y.onOneGoroutine()
	other, _ := marked.WithValues(make([]float64, marked.Len()))
	if !marked.Head(10).alone || !marked.Tail(5).alone || !marked.WithPhase(3).alone || !other.alone ||
		!marked.flat(nil).alone || y.alone || y.flat(nil).alone {
		t.Error("the mark of a series for one goroutine")
	}
	if flat := Monthly(y.Values(), 0).Slice(40, 60).flat(make([]float64, 20)); flat.Start() != 40 || flat.Period() != 1 || flat.Season(3) != 0 {
		t.Errorf("flat series %+v", flat)
	}
}

func TestEdgeHelpersWithNegativeArguments(t *testing.T) {
	v := []float64{1, 2, 4, 7, 11}
	x := Regressors{}.With("a", v)
	quiet(t, "helpers", func() {
		if !x.Covers(-1) || !x.Covers(0) || !x.Covers(5) || x.Covers(6) || !(Regressors{}).Covers(-3) {
			t.Error("covers")
		}
		for _, lag := range []int{0, -1, 5, 9} {
			if d := Difference(v, lag); len(d) != 0 {
				t.Errorf("difference at lag %d: %v", lag, d)
			}
		}
		if d := Difference(v, 2); !slices.Equal(d, []float64{3, 5, 7}) {
			t.Errorf("difference at lag 2: %v", d)
		}
		if len(ACF(v, 0)) != 0 || len(ACF(v, -2)) != 0 || len(ACF(v, 2)) != 2 || len(ACF(nil, 2)) != 2 {
			t.Error("autocorrelations")
		}
		for _, period := range []float64{0, -12, math.NaN(), math.Inf(1), math.Inf(-1)} {
			if w := Fourier(period, 3, 24).Width(); w != 0 {
				t.Errorf("fourier of period %v: %d columns", period, w)
			}
		}
		if Fourier(12, 3, -1).Width() != 0 || Fourier(12, -2, 24).Width() != 0 || Fourier(12, 3, 0).Width() != 6 {
			t.Error("fourier")
		}
		if SeasonalDummies(4, -1).Width() != 0 || SeasonalDummies(1, 8).Width() != 0 || SeasonalDummies(0, 8).Width() != 0 ||
			SeasonalDummies(-4, 8).Width() != 0 || SeasonalDummies(4, 0).Width() != 3 {
			t.Error("seasonal dummies")
		}
		if NDiffs(v, -1) != 0 || NSDiffs(v, -3) != 0 {
			t.Error("differences")
		}
		if _, ok := MASEScale(v, -2); !ok {
			t.Error("scale")
		}
		if _, ok := Interpolate(v, -5); !ok {
			t.Error("interpolation")
		}
	})
}

func TestEdgeZeroValueFits(t *testing.T) {
	quiet(t, "ArimaFit", func() {
		f := &ArimaFit{}
		if len(f.Forecast(1)) != 0 || len(f.ForecastVariance(1)) != 0 || f.Period() != 0 {
			t.Error("forecast of an empty ARIMA fit")
		}
		f.Params()
		f.IsWellBehaved(1.01)
		f.Order()
		f.SeasonalOrder()
	})
	quiet(t, "ProphetFit", func() {
		f := &ProphetFit{}
		if len(f.Forecast(1)) != 0 || len(f.Params()) != 0 || len(f.Changepoints()) != 0 || len(f.Effects()) != 0 ||
			len(f.FittedValues()) != 0 || !math.IsNaN(f.Trend(0)) || !math.IsNaN(f.Seasonal(3)) || !math.IsNaN(f.Events(-1)) ||
			!math.IsNaN(f.Sigma()) {
			t.Error("an empty Prophet fit")
		}
	})
	quiet(t, "TbatsFit", func() {
		f := &TbatsFit{}
		if len(f.Forecast(1)) != 0 || len(f.Seasonal()) != 0 || len(f.Residuals()) != 0 || len(f.InitialStates()) != 0 {
			t.Error("an empty TBATS fit")
		}
		f.Params()
		f.Lambda()
		f.Trend()
		f.Arma()
		f.Smoothing()
	})
	quiet(t, "EtsFit", func() {
		f := &EtsFit{}
		if len(f.Forecast(1)) != 0 {
			t.Error("forecast of an empty ETS fit")
		}
		f.Params()
		f.Model()
		f.InitialSeasonal()
	})
	quiet(t, "Report", func() {
		r := &Report{}
		if best := r.Best(); best == nil || best.Name != "" || len(best.Forecast) != 0 {
			t.Error("best of an empty report")
		}
		if _, ok := r.Candidate("naive"); ok {
			t.Error("candidate of an empty report")
		}
		if best := (&Report{Candidates: make([]CandidateReport, 2), Chosen: 5}).Best(); best == nil {
			t.Error("chosen out of range")
		}
		var none *Report
		if none.Best() == nil {
			t.Error("best of no report")
		}
		if _, ok := (CandidateReport{}).Cumulative(1); ok {
			t.Error("cumulative of an empty report")
		}
		if _, ok := (CandidateReport{Forecast: make([]Point, 3)}).Cumulative(2); ok {
			t.Error("cumulative without the measures of the backtest")
		}
		if _, ok := (Point{}).Interval(0.8); ok {
			t.Error("interval of an empty point")
		}
	})
	quiet(t, "Decomposition", func() {
		var d Decomposition
		if len(d.SeasonallyAdjusted()) != 0 {
			t.Error("an empty decomposition")
		}
		d.TrendStrength()
		if _, ok := d.SeasonalStrength(0); ok {
			t.Error("seasonal strength of an empty decomposition")
		}
	})
}
