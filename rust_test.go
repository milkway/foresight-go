package foresight

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
)

// The Rust crate, through the Radar Fiscal project, recorded the backtest of
// four models on two public series. The same models here have to give the
// same numbers.
type rustQuantiles struct {
	Q025 float64 `json:"q025"`
	Q10  float64 `json:"q10"`
	Q90  float64 `json:"q90"`
	Q975 float64 `json:"q975"`
	N    int     `json:"n"`
}

type rustResult struct {
	Chosen     string `json:"escolhido"`
	Candidates []struct {
		Name       string           `json:"modelo"`
		MAPE       []float64        `json:"mape"`
		Bias       []float64        `json:"vies"`
		N          []int            `json:"n"`
		Score      float64          `json:"mape_medio"`
		Quantiles  []*rustQuantiles `json:"quantis"`
		Cumulative []*rustQuantiles `json:"quantis_acumulado"`
	} `json:"candidatos"`
	Forecasts []struct {
		Name   string `json:"modelo"`
		Points []struct {
			Mean float64 `json:"previsto"`
			Lo80 float64 `json:"li80"`
			Hi80 float64 `json:"ls80"`
			Lo95 float64 `json:"li95"`
			Hi95 float64 `json:"ls95"`
		} `json:"pontos"`
	} `json:"previsoes"`
	Retroactive [][]float64 `json:"retroativo"`
	Origins     int         `json:"origens"`
}

var rustNames = strings.NewReplacer(
	"media(", "mean(",
	"sazonal_ingenuo", "seasonal_naive_growth",
	"regressao_log_ipca", "log_linear_deflated",
	"regressao_log", "log_linear",
)

func same(t *testing.T, what string, ours, theirs float64) {
	t.Helper()
	if math.Abs(ours-theirs) > 1e-9*math.Max(math.Abs(theirs), 1e-3) {
		t.Errorf("%s: %v, want %v", what, ours, theirs)
	}
}

func TestSameAsRust(t *testing.T) {
	raw, err := os.ReadFile("testdata/rust_classic.json")
	if err != nil {
		t.Fatal(err)
	}
	var recorded map[string]map[string]rustResult
	if err := json.Unmarshal(raw, &recorded); err != nil {
		t.Fatal(err)
	}
	index := ipca(t)
	for name, values := range map[string][]float64{"icms": icms(t), "fpe": fpe(t)} {
		for variant, deflated := range map[string]bool{"com_ipca": true, "sem_ipca": false} {
			candidates := []Candidate{
				NewCandidate(SeasonalNaive{Growth: true}),
				NewCandidate(HoltWinters{}),
				NewCandidate(LogLinear{Window: 72}),
			}
			if deflated {
				candidates = append(candidates, NewCandidate(LogLinear{Window: 72, Deflator: index}))
			}
			report, err := DefaultBacktest().Run(Monthly(values, 2), candidates)
			if err != nil {
				t.Fatal(err)
			}
			want := recorded[name][variant]
			where := name + " " + variant
			if got := report.Best().Name; got != rustNames.Replace(want.Chosen) {
				t.Errorf("%s: chose %s, want %s", where, got, want.Chosen)
			}
			if report.Origins != want.Origins || len(report.Candidates) != len(want.Candidates) {
				t.Fatalf("%s: %d origins and %d candidates", where, report.Origins, len(report.Candidates))
			}
			for i, w := range want.Candidates {
				c := report.Candidates[i]
				at := where + " " + c.Name
				if c.Name != rustNames.Replace(w.Name) {
					t.Fatalf("%s: want %s", at, w.Name)
				}
				same(t, at+" score", c.Score, w.Score)
				for h, s := range c.Horizons {
					same(t, at+" mape", s.MAPE, w.MAPE[h])
					same(t, at+" bias", s.Bias, w.Bias[h])
					if s.N != w.N[h] {
						t.Errorf("%s: %d pairs, want %d", at, s.N, w.N[h])
					}
					for j, q := range []*rustQuantiles{w.Quantiles[h], w.Cumulative[h]} {
						b := s.Bands
						if j == 1 {
							b = s.Cumulative
						}
						same(t, at+" q10", b[0].Lower, q.Q10)
						same(t, at+" q90", b[0].Upper, q.Q90)
						same(t, at+" q025", b[1].Lower, q.Q025)
						same(t, at+" q975", b[1].Upper, q.Q975)
					}
				}
				for h, p := range want.Forecasts[i].Points {
					f := c.Forecast[h]
					same(t, at+" forecast", f.Mean, p.Mean)
					i80, _ := f.Interval(0.80)
					i95, _ := f.Interval(0.95)
					same(t, at+" li80", i80.Lower, p.Lo80)
					same(t, at+" ls80", i80.Upper, p.Hi80)
					same(t, at+" li95", i95.Lower, p.Lo95)
					same(t, at+" ls95", i95.Upper, p.Hi95)
				}
			}
			for o, row := range want.Retroactive {
				for h, v := range row {
					same(t, where+" retroactive", report.Best().Trajectories[o][h], v)
				}
			}
		}
	}
}
