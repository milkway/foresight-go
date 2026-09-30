package foresight_test

import (
	"fmt"

	foresight "github.com/milkway/foresight-go"
)

// Eight years of monthly data with growth, a yearly pattern and some noise.
func sales() []float64 {
	pattern := []float64{1.1, 0.8, 0.9, 1.0, 1.0, 0.9, 1.0, 1.0, 0.9, 1.0, 1.1, 1.3}
	out := make([]float64, 96)
	level := 100.0
	for t := range out {
		noise := float64((t*37)%11-5) * 1.5
		out[t] = level*pattern[t%12] + noise
		level += 0.5
	}
	return out
}

// Replay the past with every built-in model, take the best and forecast.
func Example() {
	y := foresight.Monthly(sales(), 0) // the first observation is in January
	report, err := foresight.DefaultBacktest().Run(y, foresight.Defaults())
	if err != nil {
		fmt.Println(err)
		return
	}
	best := report.Best()
	fmt.Printf("%s, MAPE %.1f%%\n", best.Name, best.Score)
	for _, p := range best.Forecast[:3] {
		i, _ := p.Interval(0.80)
		fmt.Printf("%d: %.1f [%.1f, %.1f]\n", p.Horizon, p.Mean, i.Lower, i.Upper)
	}
	total, _ := best.Cumulative(6)
	fmt.Printf("next six months: %.0f\n", total.Mean)
	// Output:
	// log_linear, MAPE 3.1%
	// 1: 164.2 [154.7, 172.1]
	// 2: 120.2 [113.5, 125.3]
	// 3: 136.7 [128.8, 142.6]
	// next six months: 862
}

// One model on its own.
func ExampleTheta() {
	y := []float64{12, 14, 13, 15, 17, 16, 18, 20}
	fit, err := foresight.Theta{}.Fit(foresight.NonSeasonal(y))
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, v := range fit.Forecast(3) {
		fmt.Printf("%.2f\n", v)
	}
	// Output:
	// 20.52
	// 21.04
	// 21.55
}

// A model of your own joins the backtest by implementing Model.
func ExampleCandidate() {
	candidates := []foresight.Candidate{
		foresight.NewCandidate(foresight.SeasonalNaive{}).Named("same_month", "Same month of last year"),
		foresight.NewCandidate(foresight.LogLinear{Window: 60}),
	}
	report, err := foresight.Backtest{Combine: 1}.Run(foresight.Monthly(sales(), 0), candidates)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, c := range report.Candidates {
		fmt.Printf("%s: MAPE %.1f%%\n", c.Name, c.Score)
	}
	// Output:
	// same_month: MAPE 6.6%
	// log_linear: MAPE 3.3%
}
