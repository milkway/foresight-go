# foresight-go

[![Go Reference](https://pkg.go.dev/badge/github.com/milkway/foresight-go.svg)](https://pkg.go.dev/github.com/milkway/foresight-go)
[![CI](https://github.com/milkway/foresight-go/actions/workflows/ci.yml/badge.svg)](https://github.com/milkway/foresight-go/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Dependencies: none](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](go.mod)

Time series forecasting in Go that picks its model by what would have worked.

`foresight` fits several models to a series, replays the past to see how each
would have done, chooses by out-of-sample error and reports intervals taken
from the errors actually observed. It uses the standard library only and
every result is deterministic.

This is the Go edition of the Rust crate
[foresight](https://github.com/milkway/foresight). It is written in Go, not
wrapped: no cgo, nothing to link.

**Website:** <https://milkway.github.io/foresight-go/> ·
**Reference:** <https://pkg.go.dev/github.com/milkway/foresight-go> ·
[Português](README.pt-BR.md)

## Install

```bash
go get github.com/milkway/foresight-go
```

Go 1.22 or later.

## Use

```go
import foresight "github.com/milkway/foresight-go"

// monthly data whose first observation is in March
y := foresight.Monthly(values, 2)

// replay the last 36 months, 12 months ahead, with every built-in model
report, err := foresight.DefaultBacktest().Run(y, foresight.Defaults())
if err != nil {
	log.Fatal(err)
}

best := report.Best()
fmt.Printf("%s: MAPE %.1f%%\n", best.Name, best.Score)
for _, p := range best.Forecast {
	i, _ := p.Interval(0.80)
	fmt.Printf("%2d %.0f [%.0f, %.0f]\n", p.Horizon, p.Mean, i.Lower, i.Upper)
}

// the total of the next six months, with its own interval
halfYear, _ := best.Cumulative(6)
```

One model on its own:

```go
// seasonal ARIMA on the log scale
fit, err := foresight.Log(foresight.Airline()).Fit(y)
nextYear := fit.Forecast(12)

// orders chosen from the data, inspected
auto, err := foresight.AutoArima{}.Select(foresight.Monthly(logValues, 2))
fmt.Println(auto.Order())
```

A trend that bends, with dated events:

```go
model := foresight.Prophet{Events: []foresight.Event{
	{Name: "campaign", Positions: []int{10, 34, 58, 82, 106, 130}}, // future ones included
	{Name: "new_law", Step: true, StepFrom: 80},                     // a lasting change of level
}}
fit, err := model.Estimate(y)
fmt.Println(fit.Changepoints(), fit.Effects())
```

A model of your own joins the backtest by implementing `Model`.

## What is in it

| Piece | What it does |
|---|---|
| `Series` | values + seasonal period; slices keep season and position |
| `Model` / `Fitted` | fit once, forecast any horizon, inspect parameters |
| Models | `Mean`, `Naive`, `Drift`, `SeasonalNaive`, `Theta`, `HoltWinters`, `LogLinear` (optionally deflated by a price index), `Arima` (seasonal, exact maximum likelihood, optionally with regressors), `AutoArima` (differences by tests, orders by stepwise search), `Ets` (the exponential smoothing family in state space form), `AutoEts`, `Prophet` (trend with changepoints, Fourier seasonality, dated events and steps), `Tbats` (several seasonal periods, not necessarily whole numbers), `Croston` (intermittent demand, with the SBA and TSB variants) |
| `Stl`, `Mstl`, `Decomposed` | trend, seasonal patterns and remainder by LOESS, for one or several periods; any model on the seasonally adjusted series |
| `Ensemble` | several models combined: plain average, median, weights by inverse error or stacked weights |
| `Interpolate`, `Outliers`, `Clean` | gaps filled and outliers found and replaced, with the season taken into account |
| `Transformed` | any model on the log or another Box-Cox scale (`Log`, `WithBoxCox`, `WithGuerrero`) |
| `Regressors` | external variables aligned with the data, `Fourier` terms, `SeasonalDummies` |
| `Defaults`, `Thorough` | ready sets of candidates: the models that fit in a moment, and those plus the automatic choices and the ensembles |
| `Backtest` | rolling origin (expanding or fixed window) on all cores; MAPE, MAE, RMSE, MASE and bias by horizon; average of the best models; choice by out-of-sample error |
| Intervals | empirical quantiles of the backtest errors, by horizon and for cumulative totals |
| Measures and tests | `MAPE`, `Bias`, `MAE`, `RMSE`, `MASE`, `Quantile`, `ACF`, `KPSS`, `NDiffs`, `NSDiffs`, `SeasonalStrength` |

## How it differs from the usual toolkits

Most forecasting libraries choose a model by an in-sample information
criterion and derive intervals from distributional assumptions. Here the
choice and the intervals both come from forecasts made without seeing the
future they are judged against. The interval for a total (say, the rest of a
fiscal year) is measured on totals, because adding up monthly limits
overstates its uncertainty.

## Checked

| Against | What | Agreement |
|---|---|---|
| The Rust crate | backtest of four models on two public series: errors and bias by horizon, quantiles, forecasts, intervals and the choice | the same numbers (relative difference under 10⁻⁹) |
| The Rust crate | Prophet, KPSS, strength of seasonality | the same numbers (under 10⁻⁸) |
| The Rust crate | ARIMA, regression with ARIMA errors, ETS: likelihood | the same (under 10⁻⁶) |
| The Rust crate | ARIMA and ETS forecasts; automatic orders and automatic ETS on three series | forecasts within the precision of the search; the same models chosen |
| The Rust crate | STL, MSTL, forecasts by decomposition, Croston, SBA and TSB, outliers and cleaning | the same numbers (under 10⁻⁸) |
| The Rust crate | ensembles: weights and forecasts; TBATS: structure chosen, likelihood and forecasts; backtest of 18 candidates | the same structure and the same choice; forecasts within the precision of the search |
| R package `forecast` 9.0.2 | seasonal naive, random walk with drift | exact |
| R package `forecast` 9.0.2 | Theta | forecasts within 0.1% |

The tests are in `rust_test.go`, `models_test.go`, `more_test.go` and `r_test.go`. The Rust
crate is in turn compared with the R packages `forecast` and `prophet`.

## Data

- `testdata/piaui_revenue.csv`: monthly ICMS and FPE revenue of the state of
  Piauí, Brazil, March 2017 to June 2026 (Siconfi/STN, RREO Anexo 03), with
  the chained IPCA price index (BCB/SGS 433). Public data.
- `testdata/air_passengers.csv`: the classic monthly airline passengers
  series (Box & Jenkins, 1976).

## Status

Early: the API may change before 1.0.

## Development

```bash
scripts/dev-check.sh   # gofmt, go vet, go test
```

## License

MIT. See [LICENSE](LICENSE).
