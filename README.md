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
fit, err := foresight.Theta{}.Fit(foresight.NonSeasonal(values))
next := fit.Forecast(3)
```

A model of your own joins the backtest by implementing `Model`.

## What is in it

| Piece | What it does |
|---|---|
| `Series` | values + seasonal period; slices keep season and position |
| `Model` / `Fitted` | fit once, forecast any horizon, inspect parameters |
| Models | `Mean`, `Naive`, `Drift`, `SeasonalNaive`, `Theta`, `HoltWinters`, `LogLinear` (optionally deflated by a price index) |
| `Backtest` | rolling origin (expanding or fixed window) on all cores; MAPE, MAE, RMSE, MASE and bias by horizon; average of the best models; choice by out-of-sample error |
| Intervals | empirical quantiles of the backtest errors, by horizon and for cumulative totals |
| Measures | `MAPE`, `Bias`, `MAE`, `RMSE`, `MASE`, `Quantile`, `ACF` |

The Rust crate has more: seasonal ARIMA with automatic orders, the exponential
smoothing family, Prophet, TBATS, STL and MSTL, regression with ARIMA errors,
intermittent demand, data cleaning and ensembles.

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
| R package `forecast` 9.0.2 | seasonal naive, random walk with drift | exact |
| R package `forecast` 9.0.2 | Theta | forecasts within 0.1% |

The tests are in `rust_test.go` and `r_test.go`.

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
