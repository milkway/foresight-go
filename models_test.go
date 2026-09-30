package foresight

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

// Results recorded by the Rust crate (version 0.7.1) on public data. Models
// estimated by a search have to agree within the precision of the search;
// everything else, to the last digits.
type rustModels struct {
	Airline struct {
		MA       []float64 `json:"ma"`
		SMA      []float64 `json:"sma"`
		LogLik   float64   `json:"loglik"`
		AICc     float64   `json:"aicc"`
		Forecast []float64 `json:"forecast"`
	} `json:"airline"`
	Mixed struct {
		AR       []float64 `json:"ar"`
		MA       []float64 `json:"ma"`
		SAR      []float64 `json:"sar"`
		Constant float64   `json:"constant"`
		LogLik   float64   `json:"loglik"`
		Forecast []float64 `json:"forecast"`
	} `json:"mixed"`
	AutoArima struct {
		Order    []int     `json:"order"`
		Constant bool      `json:"constant"`
		AICc     float64   `json:"aicc"`
		Forecast []float64 `json:"forecast"`
	} `json:"auto_arima"`
	Ets map[string]struct {
		LogLik   float64   `json:"loglik"`
		AICc     float64   `json:"aicc"`
		Alpha    float64   `json:"alpha"`
		Forecast []float64 `json:"forecast"`
	} `json:"ets"`
	AutoEts struct {
		Code string  `json:"code"`
		AICc float64 `json:"aicc"`
	} `json:"auto_ets"`
	Prophet    rustProphet `json:"prophet"`
	Prophet5   rustProphet `json:"prophet5"`
	ProphetLog rustProphet `json:"prophet_log"`
	Guerrero   float64     `json:"guerrero"`
	KPSS       float64     `json:"kpss"`
	NDiffs     int         `json:"ndiffs"`
	NSDiffs    int         `json:"nsdiffs"`
	Strength   float64     `json:"strength"`
	Backtest   struct {
		Chosen   string            `json:"chosen"`
		Scores   []json.RawMessage `json:"scores"`
		Forecast []float64         `json:"forecast"`
	} `json:"backtest"`
}

type rustProphet struct {
	Forecast []float64 `json:"forecast"`
	Bends    []float64 `json:"bends"`
	Sigma    float64   `json:"sigma"`
}

func recorded(t *testing.T) map[string]json.RawMessage {
	t.Helper()
	raw, err := os.ReadFile("testdata/rust_models.json")
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]json.RawMessage
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func series(t *testing.T, name string) (values []float64, first int) {
	switch name {
	case "air":
		return air(t), 0
	case "icms":
		return icms(t), 2
	}
	return fpe(t), 2
}

func logs(v []float64) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = math.Log(x)
	}
	return out
}

func near(t *testing.T, what string, ours, theirs, tolerance float64) {
	t.Helper()
	if math.Abs(ours-theirs) > tolerance*math.Max(math.Abs(theirs), 1) {
		t.Errorf("%s: %v, want %v", what, ours, theirs)
	}
}

func nearAll(t *testing.T, what string, ours, theirs []float64, tolerance float64) {
	t.Helper()
	if len(ours) != len(theirs) {
		t.Fatalf("%s: %d values, want %d", what, len(ours), len(theirs))
	}
	for i := range ours {
		near(t, what, ours[i], theirs[i], tolerance)
	}
}

// what a search may differ by, and what arithmetic may
const (
	searched = 1e-5
	computed = 1e-9
)

func TestArimaSameAsRust(t *testing.T) {
	all := recorded(t)
	for _, name := range []string{"air", "icms", "fpe"} {
		var want rustModels
		if err := json.Unmarshal(all[name], &want); err != nil {
			t.Fatal(err)
		}
		v, first := series(t, name)
		y, ly := Monthly(v, first), Monthly(logs(v), first)

		a, err := Airline().Estimate(ly)
		if err != nil {
			t.Fatal(err)
		}
		// a coefficient on the edge of invertibility is flat in the likelihood
		nearAll(t, name+" airline ma", a.MA, want.Airline.MA, 1e-3)
		nearAll(t, name+" airline sma", a.SeasonalMA, want.Airline.SMA, 1e-3)
		near(t, name+" airline loglik", a.LogLikelihood, want.Airline.LogLik, 1e-7)
		near(t, name+" airline aicc", a.AICc, want.Airline.AICc, 1e-7)
		nearAll(t, name+" airline forecast", must(t, Log(Airline()), y, 12), want.Airline.Forecast, searched)

		m, err := Arima{P: 1, D: 1, Q: 1, SeasonalP: 1, Constant: WithConstant}.Estimate(ly)
		if err != nil {
			t.Fatal(err)
		}
		nearAll(t, name+" mixed ar", m.AR, want.Mixed.AR, 1e-3)
		nearAll(t, name+" mixed sar", m.SeasonalAR, want.Mixed.SAR, 1e-3)
		near(t, name+" mixed constant", m.Constant, want.Mixed.Constant, 1e-4)
		near(t, name+" mixed loglik", m.LogLikelihood, want.Mixed.LogLik, 1e-7)
		nearAll(t, name+" mixed forecast", m.Forecast(12), want.Mixed.Forecast, searched)

		auto, err := AutoArima{}.Select(ly)
		if err != nil {
			t.Fatal(err)
		}
		p, d, q := auto.Order()
		sp, sd, sq := auto.SeasonalOrder()
		for i, o := range []int{p, d, q, sp, sd, sq} {
			if o != want.AutoArima.Order[i] {
				t.Fatalf("%s: automatic orders (%d,%d,%d)(%d,%d,%d), want %v", name, p, d, q, sp, sd, sq, want.AutoArima.Order)
			}
		}
		if auto.HasConstant != want.AutoArima.Constant {
			t.Errorf("%s: constant %v", name, auto.HasConstant)
		}
		near(t, name+" auto aicc", auto.AICc, want.AutoArima.AICc, 1e-6)
		nearAll(t, name+" auto forecast", auto.Forecast(12), want.AutoArima.Forecast, 1e-4)
	}
}

func TestRegressionSameAsRust(t *testing.T) {
	all := recorded(t)
	var index, fourier struct {
		Slope    float64   `json:"slope"`
		Constant float64   `json:"constant"`
		LogLik   float64   `json:"loglik"`
		Forecast []float64 `json:"forecast"`
	}
	if json.Unmarshal(all["regression_index"], &index) != nil || json.Unmarshal(all["regression_fourier"], &fourier) != nil {
		t.Fatal("recorded results")
	}
	model := Airline()
	model.Regressors = Regressors{}.With("ipca", logs(ipca(t)))
	fit, err := model.Estimate(NewSeries(logs(icms(t))[:100], 12))
	if err != nil {
		t.Fatal(err)
	}
	near(t, "index slope", fit.Regression[0].Value, index.Slope, 1e-4)
	near(t, "index loglik", fit.LogLikelihood, index.LogLik, 1e-7)
	nearAll(t, "index forecast", fit.Forecast(12), index.Forecast, searched)

	harmonic := Arima{P: 1, D: 1, Q: 1, Constant: WithConstant, Regressors: Fourier(12, 3, 156)}
	fit, err = harmonic.Estimate(NewSeries(logs(air(t)), 12))
	if err != nil {
		t.Fatal(err)
	}
	near(t, "fourier constant", fit.Constant, fourier.Constant, 1e-4)
	near(t, "fourier loglik", fit.LogLikelihood, fourier.LogLik, 1e-7)
	nearAll(t, "fourier forecast", fit.Forecast(12), fourier.Forecast, searched)
	if harmonic.Name() != "arima_111_x" || fit.Regression[0].Name != "sin1_12" {
		t.Errorf("names %s %s", harmonic.Name(), fit.Regression[0].Name)
	}
	// the regressors have to reach the horizon
	if _, err := Forecast(harmonic, NewSeries(logs(air(t)), 12), 13); err == nil {
		t.Error("forecast beyond the rows of the regressors")
	}
}

func TestEtsSameAsRust(t *testing.T) {
	all := recorded(t)
	for _, name := range []string{"air", "icms", "fpe"} {
		var want rustModels
		if err := json.Unmarshal(all[name], &want); err != nil {
			t.Fatal(err)
		}
		v, first := series(t, name)
		y := Monthly(v, first)
		for code, w := range want.Ets {
			model, ok := EtsFromCode(code)
			if !ok || model.Code() != code {
				t.Fatalf("code %s", code)
			}
			fit, err := model.Estimate(y)
			if err != nil {
				t.Fatal(err)
			}
			at := name + " " + code
			// many numbers are searched at once; what counts is the likelihood
			near(t, at+" loglik", fit.LogLikelihood, w.LogLik, 1e-6)
			near(t, at+" aicc", fit.AICc, w.AICc, 1e-6)
			nearAll(t, at+" forecast", fit.Forecast(12), w.Forecast, 2e-3)
		}
		auto, err := AutoEts{}.Select(y)
		if err != nil {
			t.Fatal(err)
		}
		if auto.Model().Code() != want.AutoEts.Code {
			t.Errorf("%s: chose %s, want %s", name, auto.Model().Code(), want.AutoEts.Code)
		}
		near(t, name+" auto ets", auto.AICc, want.AutoEts.AICc, 1e-6)
	}
}

func TestProphetSameAsRust(t *testing.T) {
	all := recorded(t)
	for _, name := range []string{"air", "icms", "fpe"} {
		var want rustModels
		if err := json.Unmarshal(all[name], &want); err != nil {
			t.Fatal(err)
		}
		v, first := series(t, name)
		for _, c := range []struct {
			what  string
			model Prophet
			y     Series
			want  rustProphet
		}{
			{"prophet", Prophet{}, Monthly(v, first), want.Prophet},
			{"prophet5", Prophet{FourierOrder: 5}, Monthly(v, first), want.Prophet5},
			{"prophet log", Prophet{}, Monthly(logs(v), first), want.ProphetLog},
		} {
			fit, err := c.model.Estimate(c.y)
			if err != nil {
				t.Fatal(err)
			}
			at := name + " " + c.what
			nearAll(t, at, fit.Forecast(12), c.want.Forecast, 1e-8)
			near(t, at+" sigma", fit.Sigma(), c.want.Sigma, 1e-8)
			bends := fit.Changepoints()
			if len(bends) != len(c.want.Bends) {
				t.Fatalf("%s: %d changepoints, want %d", at, len(bends), len(c.want.Bends))
			}
			for i, b := range bends {
				if b.Position != int(c.want.Bends[i]) {
					t.Errorf("%s: changepoint at %d, want %v", at, b.Position, c.want.Bends[i])
				}
			}
		}
	}
	var events struct {
		Effects  []float64 `json:"effects"`
		Forecast []float64 `json:"forecast"`
	}
	if err := json.Unmarshal(all["events"], &events); err != nil {
		t.Fatal(err)
	}
	pattern := []float64{5, -3, 0, 2, -4, 1, 3, -2, 0, 4, -5, 4}
	y := make([]float64, 120)
	for i := range y {
		y[i] = 200 + 0.8*float64(i) + pattern[i%12] + float64((i*37)%11)*0.3
		if i == 10 || i == 34 || i == 58 || i == 82 || i == 106 {
			y[i] += 20
		}
		if i >= 80 {
			y[i] -= 15
		}
	}
	fit, err := Prophet{NoChangepoints: true, Events: []Event{
		{Name: "campaign", Positions: []int{10, 34, 58, 82, 106, 126}},
		{Name: "new_law", Step: true, StepFrom: 80},
	}}.Estimate(Monthly(y, 0))
	if err != nil {
		t.Fatal(err)
	}
	effects := fit.Effects()
	nearAll(t, "effects", []float64{effects[0].Value, effects[1].Value}, events.Effects, 1e-8)
	nearAll(t, "events forecast", fit.Forecast(12), events.Forecast, 1e-8)
	if math.Abs(effects[0].Value-20) > 3 || math.Abs(effects[1].Value+15) > 3 {
		t.Errorf("effects %v", effects)
	}
}

func TestDiagnosticsSameAsRust(t *testing.T) {
	all := recorded(t)
	for _, name := range []string{"air", "icms", "fpe"} {
		var want rustModels
		if err := json.Unmarshal(all[name], &want); err != nil {
			t.Fatal(err)
		}
		v, first := series(t, name)
		l := logs(v)
		stat, _ := KPSS(l)
		near(t, name+" kpss", stat, want.KPSS, computed)
		strength, _ := SeasonalStrength(l, 12)
		near(t, name+" strength", strength, want.Strength, computed)
		if NDiffs(l, 2) != want.NDiffs || NSDiffs(l, 12) != want.NSDiffs {
			t.Errorf("%s: differences %d and %d", name, NDiffs(l, 2), NSDiffs(l, 12))
		}
		lambda, _ := Guerrero(Monthly(v, first))
		near(t, name+" guerrero", lambda.Lambda, want.Guerrero, 1e-6)
	}
}

// The whole of it: the default candidates, the backtest, the choice.
func TestDefaultBacktestSameAsRust(t *testing.T) {
	all := recorded(t)
	for _, name := range []string{"air", "icms", "fpe"} {
		var want rustModels
		if err := json.Unmarshal(all[name], &want); err != nil {
			t.Fatal(err)
		}
		v, first := series(t, name)
		r, err := DefaultBacktest().Run(Monthly(v, first), Defaults())
		if err != nil {
			t.Fatal(err)
		}
		if r.Best().Name != want.Backtest.Chosen || len(r.Candidates) != len(want.Backtest.Scores) {
			t.Fatalf("%s: chose %s among %d, want %s among %d", name, r.Best().Name,
				len(r.Candidates), want.Backtest.Chosen, len(want.Backtest.Scores))
		}
		for i, raw := range want.Backtest.Scores {
			var pair []any
			if err := json.Unmarshal(raw, &pair); err != nil {
				t.Fatal(err)
			}
			c := r.Candidates[i]
			if c.Name != pair[0].(string) {
				t.Fatalf("%s: candidate %s, want %v", name, c.Name, pair[0])
			}
			near(t, name+" "+c.Name+" score", c.Score, pair[1].(float64), 1e-4)
		}
		means := make([]float64, len(r.Best().Forecast))
		for i, p := range r.Best().Forecast {
			means[i] = p.Mean
		}
		nearAll(t, name+" forecast", means, want.Backtest.Forecast, searched)
	}
}
