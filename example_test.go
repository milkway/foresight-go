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

// Seasonal ARIMA on the log scale, the model that does best on many monthly
// series.
func ExampleAirline() {
	model := foresight.Log(foresight.Airline())
	fit, err := model.Fit(foresight.Monthly(sales(), 0))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(model.Name())
	for _, v := range fit.Forecast(3) {
		fmt.Printf("%.0f\n", v)
	}
	// Output:
	// log_arima_011_011
	// 164
	// 121
	// 137
}

// A trend that bends, with an event that recurs and is known ahead.
func ExampleProphet() {
	y := sales()
	for _, t := range []int{5, 29, 53, 77} {
		y[t] += 30 // a campaign every other June
	}
	model := foresight.Prophet{Events: []foresight.Event{
		{Name: "campaign", Positions: []int{5, 29, 53, 77, 101}},
	}}
	fit, err := model.Estimate(foresight.Monthly(y, 0))
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, e := range fit.Effects() {
		fmt.Printf("%s: %+.0f\n", e.Name, e.Value)
	}
	// Output:
	// campaign: +29
}

// The member of the exponential smoothing family that suits the series best.
func ExampleAutoEts() {
	fit, err := foresight.AutoEts{}.Select(foresight.Monthly(sales(), 0))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("ETS(%s)\n", fit.Model().Code())
	// Output:
	// ETS(MAM)
}

// Trend, seasonal pattern and what is left.
func ExampleStl() {
	d, err := foresight.Stl{Period: 12}.Decompose(sales())
	if err != nil {
		fmt.Println(err)
		return
	}
	strength, _ := d.SeasonalStrength(0)
	fmt.Printf("trend %.0f to %.0f, seasonal strength %.2f\n", d.Trend[0], d.Trend[95], strength)
	// Output:
	// trend 99 to 149, seasonal strength 0.89
}

// Several models combined, the better ones counting more: here the three
// that would have forecast the last year best.
func ExampleEnsemble() {
	model := foresight.Ensemble{Members: foresight.Defaults(), Top: 3}
	fit, err := model.Fit(foresight.Monthly(sales(), 0))
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, p := range fit.Params() {
		fmt.Println(p.Name)
	}
	// Output:
	// weight_log_linear
	// weight_log_prophet
	// weight_holt_winters
}

// Demand that comes now and then: the rate per period.
func ExampleCroston() {
	y := []float64{0, 0, 3, 0, 0, 0, 2, 0, 0, 4, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0}
	fit, err := foresight.Croston{Variant: foresight.SBA}.Fit(foresight.NonSeasonal(y))
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%.2f per period\n", fit.Forecast(1)[0])
	// Output:
	// 0.88 per period
}

// A value that does not belong, found and replaced.
func ExampleOutliers() {
	y := sales()
	y[40] *= 3
	found, _ := foresight.Outliers(y, 12)
	for _, o := range found {
		fmt.Printf("position %d: %.0f, rather %.0f\n", o.Index, o.Value, o.Replacement)
	}
	// Output:
	// position 40: 364, rather 121
}
