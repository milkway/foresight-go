package foresight

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// The results of the Rust crate for decomposition, intermittent demand,
// cleaning, ensembles and TBATS, recorded in testdata/rust_more.json.

func more(t *testing.T, section string, into any) {
	t.Helper()
	raw, err := os.ReadFile("testdata/rust_more.json")
	if err != nil {
		t.Fatal(err)
	}
	var all map[string]json.RawMessage
	if err := json.Unmarshal(raw, &all); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(all[section], into); err != nil {
		t.Fatal(err)
	}
}

func TestStlSameAsRust(t *testing.T) {
	var want map[string]struct {
		Seasonal         []float64 `json:"seasonal"`
		Trend            []float64 `json:"trend"`
		TrendStrength    float64   `json:"trend_strength"`
		SeasonalStrength float64   `json:"seasonal_strength"`
	}
	more(t, "stl", &want)
	y := logs(air(t))
	cases := map[string]Stl{
		"periodic":      {Period: 12},
		"span13":        {Period: 12, SeasonalWindow: 13},
		"robust7":       {Period: 12, SeasonalWindow: 7, Robust: true},
		"robust5rounds": {Period: 12, SeasonalWindow: 7, Robust: true, Inner: 1, Outer: 5},
		"degree1":       {Period: 12, SeasonalWindow: 11, SeasonalLinear: true, TrendWindow: 21},
		"flat":          {Period: 12, SeasonalWindow: 9, FlatTrend: true, FlatLowPass: true},
	}
	for name, stl := range cases {
		d, err := stl.Decompose(y)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		nearAll(t, name+" seasonal", d.Seasonal[0], want[name].Seasonal, computed)
		nearAll(t, name+" trend", d.Trend, want[name].Trend, computed)
		near(t, name+" trend strength", d.TrendStrength(), want[name].TrendStrength, computed)
		s, _ := d.SeasonalStrength(0)
		near(t, name+" seasonal strength", s, want[name].SeasonalStrength, computed)
		for i := range y {
			near(t, name+" sum", d.Trend[i]+d.Seasonal[0][i]+d.Remainder[i], y[i], 1e-12)
		}
	}
}

// twoPatterns is a daily series with a weekly and a monthly pattern.
func twoPatterns() []float64 {
	weekly := []float64{5, 0, -2, -3, 0, 1, -1}
	out := make([]float64, 420)
	for t := range out {
		x := float64(t)
		out[t] = 100 + 0.05*x + weekly[t%7] + 8*math.Sin(2*math.Pi*x/30) + 2*math.Sin(x*1.7)*math.Cos(x*0.3)
	}
	return out
}

func TestMstlSameAsRust(t *testing.T) {
	var want struct {
		Seasonal7  []float64 `json:"seasonal7"`
		Seasonal30 []float64 `json:"seasonal30"`
		Trend      []float64 `json:"trend"`
	}
	more(t, "mstl", &want)
	d, err := Mstl{Periods: []int{30, 7}}.Decompose(twoPatterns())
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Periods) != 2 || d.Periods[0] != 7 || d.Periods[1] != 30 {
		t.Fatalf("periods %v", d.Periods)
	}
	// the sines of the series differ in the last digits between platforms
	nearAll(t, "weekly", d.Seasonal[0], want.Seasonal7, 1e-8)
	nearAll(t, "monthly", d.Seasonal[1], want.Seasonal30, 1e-8)
	nearAll(t, "trend", d.Trend, want.Trend, 1e-8)
}

func TestDecomposedSameAsRust(t *testing.T) {
	var want struct {
		Naive []float64 `json:"icms_naive"`
		Drift []float64 `json:"icms_drift"`
		Two   []float64 `json:"two"`
	}
	more(t, "decomposed", &want)
	y := NewSeries(icms(t), 12)
	naive, err := Forecast(Decomposed{Model: Naive{}}, y, 12)
	if err != nil {
		t.Fatal(err)
	}
	nearAll(t, "naive", naive, want.Naive, computed)
	drift, err := Forecast(Decomposed{Model: Drift{}}, y, 12)
	if err != nil {
		t.Fatal(err)
	}
	nearAll(t, "drift", drift, want.Drift, computed)
	two, err := Forecast(Decomposed{Model: Drift{}, Periods: []int{7, 30}}, NewSeries(twoPatterns(), 7), 40)
	if err != nil {
		t.Fatal(err)
	}
	nearAll(t, "two periods", two, want.Two, 1e-8)
	if name := (Decomposed{Model: Drift{}}).Name(); name != "stl_drift" {
		t.Errorf("name %q", name)
	}
}

func demand() []float64 {
	out := make([]float64, 60)
	for t := range out {
		if (t*7)%10 < 3 {
			out[t] = float64(1 + (t*3)%5)
		}
	}
	return out
}

func TestCrostonSameAsRust(t *testing.T) {
	var want map[string]float64
	more(t, "croston", &want)
	y := NonSeasonal(demand())
	cases := map[string]Croston{
		"croston":  {},
		"croston3": {Alpha: 0.3},
		"sba":      {Variant: SBA, Alpha: 0.2},
		"tsb":      {Variant: TSB, Alpha: 0.2, Beta: 0.1},
	}
	for name, model := range cases {
		f, err := Forecast(model, y, 3)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		near(t, name, f[0], want[name], computed)
		if f[2] != f[0] {
			t.Errorf("%s: the rate changes with the horizon", name)
		}
	}
	f, err := Forecast(Croston{Optimised: true}, y, 1)
	if err != nil {
		t.Fatal(err)
	}
	near(t, "optimised", f[0], want["optimised"], searched)
}

func TestCrostonRefusesNegativeValues(t *testing.T) {
	if _, err := (Croston{}).Fit(NonSeasonal([]float64{1, 0, -2, 0, 3, 0, 0, 1})); err == nil {
		t.Error("negative demand accepted")
	}
	// no demand at all: nothing is expected
	f, err := Forecast(Croston{}, NonSeasonal(make([]float64, 20)), 2)
	if err != nil || f[0] != 0 || f[1] != 0 {
		t.Errorf("a series without demand: %v, %v", f, err)
	}
}

func TestCleanSameAsRust(t *testing.T) {
	var want map[string]struct {
		Index       []float64 `json:"index"`
		Replacement []float64 `json:"replacement"`
		Clean       []float64 `json:"clean"`
		Filled      []float64 `json:"filled"`
	}
	more(t, "clean", &want)
	dirty := logs(air(t))
	dirty[29] += 0.8
	dirty[99] -= 0.7
	dirty[60], dirty[61] = math.NaN(), math.NaN()
	doubled := append([]float64(nil), icms(t)...)
	doubled[39] *= 2
	doubled[89] *= 0.4
	cases := []struct {
		name   string
		values []float64
		period int
	}{{"air", dirty, 12}, {"icms", doubled, 12}, {"fpe", fpe(t), 12}, {"air_flat", dirty, 1}}
	for _, c := range cases {
		found, ok := Outliers(c.values, c.period)
		if !ok {
			t.Fatalf("%s: no answer", c.name)
		}
		w := want[c.name]
		if len(found) != len(w.Index) {
			t.Fatalf("%s: %d outliers, want %d", c.name, len(found), len(w.Index))
		}
		for i, o := range found {
			if o.Index != int(w.Index[i]) {
				t.Errorf("%s: outlier at %d, want %v", c.name, o.Index, w.Index[i])
			}
			near(t, c.name+" replacement", o.Replacement, w.Replacement[i], computed)
		}
		cleaned, _ := Clean(c.values, c.period)
		nearAll(t, c.name+" clean", cleaned, w.Clean, computed)
		filled, _ := Interpolate(c.values, c.period)
		nearAll(t, c.name+" filled", filled, w.Filled, computed)
	}
}

func TestEnsembleSameAsRust(t *testing.T) {
	type combination struct {
		Names    []string  `json:"names"`
		Weights  []float64 `json:"weights"`
		Forecast []float64 `json:"forecast"`
	}
	var want map[string]map[string]combination
	more(t, "ensemble", &want)
	cases := map[string]Ensemble{
		"inverse_error": {},
		"equal":         {Weighting: EqualWeights},
		"median":        {Weighting: Median},
		"stacked":       {Weighting: Stacked},
		"top3":          {Top: 3},
	}
	for _, name := range []string{"icms", "fpe"} {
		values, first := series(t, name)
		y := Monthly(values, first)
		for key, e := range cases {
			e.Members = Defaults()
			fit, err := e.Fit(y)
			if err != nil {
				t.Fatalf("%s %s: %v", name, key, err)
			}
			w := want[name][key]
			params := fit.Params()
			if len(params) != len(w.Names) {
				t.Fatalf("%s %s: %d members, want %d", name, key, len(params), len(w.Names))
			}
			for i, p := range params {
				if p.Name != w.Names[i] {
					t.Errorf("%s %s: member %q, want %q", name, key, p.Name, w.Names[i])
				}
				if math.Abs(p.Value-w.Weights[i]) > 1e-4 {
					t.Errorf("%s %s: weight of %s %v, want %v", name, key, p.Name, p.Value, w.Weights[i])
				}
			}
			nearAll(t, name+" "+key, fit.Forecast(12), w.Forecast, searched)
		}
	}
}

func TestEnsembleWeightsAddUpToOne(t *testing.T) {
	y := Monthly(icms(t), 2)
	for _, w := range []Weighting{InverseError, EqualWeights, Stacked} {
		fit, err := Ensemble{Members: Defaults(), Weighting: w}.Fit(y)
		if err != nil {
			t.Fatal(err)
		}
		total := 0.0
		for _, p := range fit.Params() {
			if p.Value < 0 {
				t.Errorf("negative weight %v", p.Value)
			}
			total += p.Value
		}
		near(t, "total", total, 1, 1e-12)
	}
	if _, err := (Ensemble{}).Fit(y); err == nil {
		t.Error("an ensemble of nothing accepted")
	}
}

func TestTbatsSameAsRust(t *testing.T) {
	if testing.Short() {
		t.Skip("TBATS takes seconds")
	}
	type result struct {
		Likelihood float64   `json:"likelihood"`
		AIC        float64   `json:"aic"`
		Harmonics  int       `json:"harmonics"`
		Trend      bool      `json:"trend"`
		Damped     bool      `json:"damped"`
		BoxCox     bool      `json:"box_cox"`
		Arma       []int     `json:"arma"`
		Forecast   []float64 `json:"forecast"`
	}
	var want map[string]map[string]result
	more(t, "tbats", &want)
	cases := map[string]Tbats{
		"plain":  {Harmonics: []int{3}, BoxCox: No, Trend: Yes, Damped: No, ArmaErrors: No},
		"boxcox": {Harmonics: []int{3}, BoxCox: Yes, Trend: Yes, Damped: Yes, ArmaErrors: No},
		"arma":   {Harmonics: []int{2}, BoxCox: No, Trend: No, FixOrders: true, P: 1, Q: 1},
		"auto":   {},
	}
	for _, name := range []string{"icms", "fpe"} {
		values, _ := series(t, name)
		y := NewSeries(values, 12)
		for key, model := range cases {
			fit, err := model.Select(y)
			if err != nil {
				t.Fatalf("%s %s: %v", name, key, err)
			}
			w := want[name][key]
			_, trend := fit.Trend()
			damping, _ := fit.Trend()
			_, boxCox := fit.Lambda()
			p, q := fit.Arma()
			if h := fit.Seasonal()[0].Harmonics; h != w.Harmonics || trend != w.Trend ||
				(trend && damping < 1) != w.Damped || boxCox != w.BoxCox || p != w.Arma[0] || q != w.Arma[1] {
				t.Errorf("%s %s: harmonics %d trend %v damping %v Box-Cox %v ARMA(%d,%d), want %+v",
					name, key, h, trend, damping, boxCox, p, q, w)
				continue
			}
			// the likelihood is flat around its peak: the searches of the two
			// editions stop at points that differ more than their values
			near(t, name+" "+key+" likelihood", fit.Likelihood(), w.Likelihood, 1e-6)
			near(t, name+" "+key+" AIC", fit.AIC(), w.AIC, 1e-6)
			nearAll(t, name+" "+key+" forecast", fit.Forecast(12), w.Forecast, 5e-3)
		}
	}
}

// The whole of it, with everything the package has but TBATS.
func TestThoroughBacktestSameAsRust(t *testing.T) {
	if testing.Short() {
		t.Skip("the thorough backtest takes seconds")
	}
	var want map[string]struct {
		Chosen   string            `json:"chosen"`
		Scores   []json.RawMessage `json:"scores"`
		Forecast []float64         `json:"forecast"`
	}
	more(t, "thorough", &want)
	for _, name := range []string{"icms", "fpe"} {
		v, first := series(t, name)
		r, err := DefaultBacktest().Run(Monthly(v, first), Thorough())
		if err != nil {
			t.Fatal(err)
		}
		w := want[name]
		if r.Best().Name != w.Chosen || len(r.Candidates) != len(w.Scores) {
			t.Fatalf("%s: chose %s among %d, want %s among %d", name, r.Best().Name,
				len(r.Candidates), w.Chosen, len(w.Scores))
		}
		for i, raw := range w.Scores {
			var pair []any
			if err := json.Unmarshal(raw, &pair); err != nil {
				t.Fatal(err)
			}
			c := r.Candidates[i]
			if c.Name != pair[0].(string) {
				t.Fatalf("%s: candidate %s, want %v", name, c.Name, pair[0])
			}
			near(t, name+" "+c.Name+" score", c.Score, pair[1].(float64), 1e-3)
		}
		means := make([]float64, len(r.Best().Forecast))
		for i, p := range r.Best().Forecast {
			means[i] = p.Mean
		}
		nearAll(t, name+" forecast", means, w.Forecast, 1e-4)
	}
}
