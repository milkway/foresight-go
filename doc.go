// Package foresight forecasts time series and picks its model by what would
// have worked.
//
// It fits several models to a series, replays the past to see how each would
// have done (a rolling-origin backtest), chooses by out-of-sample error and
// reports intervals taken from the errors actually observed, including
// intervals for the total of the next k periods. It has no dependencies
// outside the standard library and every result is deterministic.
//
//	y := foresight.Monthly(values, 2) // first observation in March
//	report, err := foresight.DefaultBacktest().Run(y, foresight.Defaults())
//	if err != nil {
//		log.Fatal(err)
//	}
//	best := report.Best()
//	for _, p := range best.Forecast {
//		i, _ := p.Interval(0.80)
//		fmt.Println(p.Horizon, p.Mean, i.Lower, i.Upper)
//	}
//	total, _ := best.Cumulative(6) // the next six periods, with its own interval
//
// A single model is used through the [Model] interface:
//
//	fit, err := foresight.Theta{}.Fit(foresight.NonSeasonal(values))
//	next := fit.Forecast(3)
//
// Besides the models there are decompositions ([Stl], [Mstl]), combinations
// of models ([Ensemble]) and the cleaning of gaps and outliers ([Clean]).
//
// This is the Go edition of the Rust crate of the same name
// (https://crates.io/crates/foresight). The two are checked against each
// other and against the R packages forecast and prophet on public data.
package foresight
