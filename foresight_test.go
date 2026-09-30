package foresight

import (
	"errors"
	"math"
	"slices"
	"testing"
)

var pattern = [12]float64{1.05, 0.80, 0.95, 1.00, 0.98, 0.97, 1.02, 1.00, 0.96, 1.01, 0.96, 1.30}

// noisySeasonalGrowth: level 100 growing 0.5% a period, known multiplicative
// seasonality, and deterministic multiplicative noise uniform in ±noise/2.
func noisySeasonalGrowth(n int, noise float64) []float64 {
	seed := uint64(42)
	out := make([]float64, n)
	for t := range out {
		seed = seed*6364136223846793005 + 1442695040888963407
		u := float64(seed>>11)/float64(uint64(1)<<53) - 0.5
		out[t] = 100 * math.Pow(1.005, float64(t)) * pattern[t%12] * (1 + noise*u)
	}
	return out
}

func seasonalGrowth(n int) []float64 { return noisySeasonalGrowth(n, 0) }

func must(t *testing.T, m Model, y Series, h int) []float64 {
	t.Helper()
	p, err := Forecast(m, y, h)
	if err != nil {
		t.Fatalf("%s: %v", m.Name(), err)
	}
	return p
}

func TestSeriesSlicingKeepsSeasonAndPosition(t *testing.T) {
	v := make([]float64, 30)
	for i := range v {
		v[i] = float64(i)
	}
	y := Monthly(v, 2) // starts in March
	if y.Season(0) != 2 || y.Season(10) != 0 {
		t.Errorf("seasons %d %d", y.Season(0), y.Season(10))
	}
	s := y.Slice(10, 20)
	if s.Len() != 10 || s.Start() != 10 || s.Index(3) != 13 || s.Season(0) != 0 {
		t.Errorf("slice %+v", s)
	}
	if got := s.Tail(4); got.Season(0) != 6 || !slices.Equal(got.Values(), v[16:20]) {
		t.Errorf("tail %+v", got)
	}
	if y.Head(12).Season(12) != 2 {
		t.Error("season of the first forecast")
	}
	if q := NewSeries(v, 4).Slice(5, 20).WithPhase(3); q.Season(0) != 3 || q.Season(1) != 0 {
		t.Error("phase of a view")
	}
	if y.Head(100).Len() != 30 || y.Slice(40, 50).Len() != 0 || NewSeries(v, 0).Period() != 1 {
		t.Error("clamping")
	}
	if _, err := y.WithValues(v[:3]); !errors.Is(err, ErrLength) {
		t.Error("length")
	}
	if !y.IsFinite() || y.IsPositive() || NonSeasonal([]float64{1, math.NaN()}).IsFinite() {
		t.Error("finite and positive")
	}
}

func TestQuantileAndMeasures(t *testing.T) {
	v := []float64{5, 1, 3, 2, 4}
	for p, want := range map[float64]float64{0: 1, 0.5: 3, 1: 5, 0.1: 1.4} {
		if got, _ := Quantile(v, p); math.Abs(got-want) > 1e-12 {
			t.Errorf("quantile %v: %v", p, got)
		}
	}
	if _, ok := Quantile(nil, 0.5); ok {
		t.Error("quantile of nothing")
	}
	actual, forecast := []float64{100, 200, 0}, []float64{110, 180, 5}
	// the zero actual is left out of the percentage measures
	if math.Abs(MAPE(actual, forecast)-10) > 1e-12 || math.Abs(Bias(actual, forecast)) > 1e-12 {
		t.Error("percentage measures")
	}
	if math.Abs(MAE(actual, forecast)-35.0/3) > 1e-12 || math.Abs(RMSE(actual, forecast)-math.Sqrt(525.0/3)) > 1e-12 {
		t.Error("absolute measures")
	}
	if !math.IsNaN(MAPE(nil, nil)) {
		t.Error("no pairs")
	}
	if s, ok := MASEScale([]float64{1, 2, 3, 5, 4, 7}, 2); !ok || s != 2 || MASE([]float64{10}, []float64{13}, 2) != 1.5 {
		t.Error("scaled error")
	}
	if _, ok := MASEScale([]float64{1, 1, 1}, 1); ok {
		t.Error("zero scale")
	}
}

func TestBenchmarks(t *testing.T) {
	y := NonSeasonal([]float64{2, 4, 6, 8})
	if !slices.Equal(must(t, Mean{}, y, 2), []float64{5, 5}) ||
		!slices.Equal(must(t, Naive{}, y, 2), []float64{8, 8}) ||
		!slices.Equal(must(t, Drift{}, y, 3), []float64{10, 12, 14}) {
		t.Error("mean, naive or drift")
	}
	if _, err := (Naive{}).Fit(NonSeasonal(nil)); !errors.Is(err, ErrTooShort) {
		t.Error("empty series")
	}
	q := NewSeries([]float64{1, 2, 3, 4, 10, 20, 30, 40}, 4)
	if !slices.Equal(must(t, SeasonalNaive{}, q, 6), []float64{10, 20, 30, 40, 10, 20}) {
		t.Error("seasonal naive")
	}
	v := make([]float64, 24)
	for i := range v {
		v[i] = 10
		if i >= 12 {
			v[i] = 11
		}
	}
	p := must(t, SeasonalNaive{Growth: true}, NewSeries(v, 12), 13)
	if math.Abs(p[0]-12.1) > 1e-9 || math.Abs(p[12]-11*1.1*1.1) > 1e-9 {
		t.Errorf("growth %v", p)
	}
	if _, err := (SeasonalNaive{Growth: true}).Fit(NewSeries(v[:20], 12)); !errors.Is(err, ErrTooShort) {
		t.Error("short for growth")
	}
}

func TestHoltWintersRecoversAKnownPattern(t *testing.T) {
	all := seasonalGrowth(132)
	fit, err := HoltWinters{}.Fit(NewSeries(all[:120], 12))
	if err != nil {
		t.Fatal(err)
	}
	p := fit.Forecast(12)
	for k, v := range p {
		if math.Abs(v/all[120+k]-1) >= 0.01 {
			t.Errorf("h=%d: %v, want %v", k+1, v, all[120+k])
		}
	}
	// December is the peak and February the trough, as in the generator
	if slices.Max(p) != p[11] || slices.Min(p) != p[1] {
		t.Error("peak and trough")
	}
	for _, y := range []Series{NewSeries(all[:30], 12), NonSeasonal(all)} {
		if _, err := (HoltWinters{}).Fit(y); err == nil {
			t.Error("short or non-seasonal series accepted")
		}
	}
}

func TestLogLinear(t *testing.T) {
	all := seasonalGrowth(108)
	fit, err := LogLinear{}.Fit(NewSeries(all[:96], 12))
	if err != nil {
		t.Fatal(err)
	}
	for k, v := range fit.Forecast(12) {
		if math.Abs(v/all[96+k]-1) >= 1e-6 {
			t.Errorf("h=%d", k+1)
		}
	}
	if p := fit.Params(); p[0] != (Param{"window", 72}) || math.Abs(p[1].Value-(math.Pow(1.005, 12)-1)*100) > 1e-6 {
		t.Errorf("params %v", p)
	}
	// constant real values with seasonality; prices rise 0.4% a period
	index := make([]float64, 108)
	nominal := make([]float64, 108)
	for i := range index {
		index[i] = math.Pow(1.004, float64(i))
		nominal[i] = all[i] / math.Pow(1.005, float64(i)) * index[i]
	}
	model := LogLinear{Deflator: index}
	for _, y := range []Series{NewSeries(nominal[:96], 12), NewSeries(nominal, 12).Slice(10, 96)} {
		for k, v := range must(t, model, y, 12) {
			if math.Abs(v/nominal[96+k]-1) >= 1e-6 {
				t.Errorf("deflated h=%d", k+1)
			}
		}
	}
	if _, err := (LogLinear{Deflator: index[:50]}).Fit(NewSeries(nominal[:96], 12)); err == nil {
		t.Error("index shorter than the training data accepted")
	}
}

func TestThetaOnALine(t *testing.T) {
	y := make([]float64, 50)
	for i := range y {
		y[i] = 10 + 2*float64(i)
	}
	p := must(t, Theta{}, NonSeasonal(y), 3)
	// the level follows the data and the drift is half the slope
	if math.Abs(p[1]-p[0]-1) > 1e-9 || p[0] <= y[49] || p[0] >= y[49]+2 {
		t.Errorf("forecast %v", p)
	}
}

func classic() []Candidate {
	return []Candidate{
		NewCandidate(SeasonalNaive{Growth: true}),
		NewCandidate(HoltWinters{}),
		NewCandidate(LogLinear{}),
	}
}

func TestBacktest(t *testing.T) {
	r, err := DefaultBacktest().Run(Monthly(seasonalGrowth(112), 2), classic())
	if err != nil {
		t.Fatal(err)
	}
	if r.Origins != 36 || r.FirstOrigin != 76 || r.Horizon != 12 || len(r.Candidates) != 4 {
		t.Errorf("report %+v", r)
	}
	hw, _ := r.Candidate("holt_winters")
	if hw.Horizons[0].N != 36 || hw.Horizons[11].N != 25 || hw.Score >= 2 {
		t.Errorf("holt-winters %+v", hw.Horizons[0])
	}
	if ll, _ := r.Candidate("log_linear"); ll.Score >= 1e-6 || r.Best().Score >= 1e-6 {
		t.Error("exact models")
	}

	y := Monthly(noisySeasonalGrowth(112, 0.08), 0)
	r, err = DefaultBacktest().Run(y, classic())
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range r.Candidates {
		if c.Score < r.Best().Score {
			t.Error("not the lowest score")
		}
	}
	for _, p := range r.Best().Forecast {
		i80, _ := p.Interval(0.80)
		i95, _ := p.Interval(0.95)
		if !(i95.Lower <= i80.Lower && i80.Lower <= i80.Upper && i80.Upper <= i95.Upper) {
			t.Errorf("h=%d: intervals not nested", p.Horizon)
		}
	}
	// the total of six periods, with an interval tighter than adding limits
	total, ok := r.Best().Cumulative(6)
	width := 0.0
	for _, p := range r.Best().Forecast[:6] {
		i, _ := p.Interval(0.80)
		width += i.Upper - i.Lower
	}
	if i, _ := total.Interval(0.80); !ok || i.Upper-i.Lower >= width {
		t.Error("cumulative interval")
	}
	if _, ok := r.Best().Cumulative(13); ok {
		t.Error("beyond the horizon")
	}
	// on one core or on all, the same report
	b := DefaultBacktest()
	b.Sequential = true
	one, _ := b.Run(y, classic())
	if one.Best().Score != r.Best().Score || !slices.Equal(one.Best().Trajectories[7], r.Best().Trajectories[7]) {
		t.Error("sequential and parallel differ")
	}
}

func TestBacktestEdges(t *testing.T) {
	y := seasonalGrowth(112)
	if _, err := DefaultBacktest().Run(Monthly(y[:55], 0), classic()); !errors.Is(err, ErrFewOrigins) {
		t.Error("55 observations leave 7 origins")
	}
	if _, err := DefaultBacktest().Run(Monthly(y, 0), nil); !errors.Is(err, ErrNoCandidate) {
		t.Error("no candidates")
	}
	// a model that cannot be fitted at every origin is left out
	r, err := Backtest{MinTrain: 12, Origins: 24}.Run(Monthly(y[:48], 0),
		[]Candidate{NewCandidate(Naive{}), NewCandidate(HoltWinters{})})
	if err != nil || len(r.Candidates) != 1 || r.Best().Name != "naive" {
		t.Errorf("report %+v, %v", r, err)
	}
	// a fixed window, another metric, no average, other levels, other names
	r, err = Backtest{Window: 60, Metric: RankByMASE, Combine: 1, Levels: []float64{0.5}}.Run(
		Monthly(noisySeasonalGrowth(112, 0.05), 0),
		[]Candidate{
			NewCandidate(SeasonalNaive{}).Named("same_month", "Same month of last year"),
			NewCandidate(LogLinear{}),
		})
	if err != nil || len(r.Candidates) != 2 {
		t.Fatalf("report %+v, %v", r, err)
	}
	c, _ := r.Candidate("same_month")
	if _, ok := c.Forecast[0].Interval(0.5); !ok || c.Description != "Same month of last year" || math.IsNaN(c.Horizons[0].MASE) {
		t.Errorf("candidate %+v", c)
	}
	// quarterly data with the default candidates
	q := noisySeasonalGrowth(80, 0.05)
	r, err = Backtest{Origins: 12, Horizon: 4, MinTrain: 24}.Run(Quarterly(q, 0), Defaults())
	if err != nil || len(r.Candidates) != len(Defaults())+1 || len(r.Best().Forecast) != 4 {
		t.Errorf("quarterly %v", err)
	}
}
