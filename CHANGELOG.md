# Changelog

## v0.3.1

Fixes found in a review of the Rust crate (its version 0.7.2) and of this
edition. Results on ordinary data do not change: the comparisons with the
results recorded by the crate pass as before.

The same fixes as the crate:

- **Backtest.** A candidate that cannot give its final forecast (it cannot be
  fitted on the whole series, or a forecast is not finite) is left out of the
  report; it used to make `Run` return an error. The final forecast follows
  `Window`, like every origin. Levels outside (0, 1) are refused with
  `ErrLevels`. Intervals of negative forecasts come in order (lower ≤ upper).
- **Threads.** `SetMaxThreads` limits the goroutines of backtests and
  ensembles, and `MaxThreads` tells the limit. An ensemble inside a parallel
  backtest no longer starts goroutines of its own, and `Backtest.Sequential`
  now holds for the ensembles among its candidates.
- **`Forecast`** returns `ErrForecast` when a forecast is not finite:
  regressors that stop short of the horizon inside `Transformed`, `Decomposed`
  or an ensemble, a Box-Cox scale that cannot be brought back.
- **Prophet** fits seasonal terms from two full cycles on; shorter histories
  gave wild forecasts.
- **ARIMA.** An exact fit has finite criteria (a very large likelihood stands
  for the unbounded one), and `AutoArima` keeps it instead of discarding it: a
  constant series is forecast as that constant, a straight line as the line.
- **`Decomposed`** keeps the position of the series, so models that go by
  position (regressors, events, a deflator) stay aligned on slices and in
  windowed backtests.
- **STL, robust.** On data the fit reproduces exactly, rounding noise no
  longer decides which points count; `Outliers` reports nothing on exact data
  and still finds a real outlier in it.
- **TBATS** recognises an exact fit: a constant series takes milliseconds.
- `SeasonalNaive{Growth: true}` refuses a last cycle that does not add up to a
  positive number; `Croston` checks `Beta` only for TSB; `Fourier` leaves out
  harmonics beyond half the period.

Only in this edition:

- **No panics.** A negative horizon is an error from `Forecast`
  (`ErrHorizon`) and gives no forecasts from a fit. Invalid configurations
  are an error from `Fit` (`ErrConfig`): negative ARIMA orders, negative
  orders of the TBATS errors, components of `Ets`, weightings of `Ensemble`
  and variants of `Croston` that are not the named ones, `Log(nil)`, a
  `Decomposed` or an ensemble member without a model. The zero `Series{}` is
  an empty series without seasonality. The zero values of `ArimaFit`,
  `EtsFit`, `ProphetFit`, `TbatsFit` and `Report` answer with nothing.
  `Quantile` reports false for a p that is not a number. `Covers`,
  `Difference`, `ACF`, `Fourier` and `SeasonalDummies` take negative
  arguments.
- **Backtest.** A model that panics is left out of the report instead of
  taking the process down from a worker, and so is a candidate without a
  model. A candidate whose forecasts are not numbers can no longer win: it is
  left out, and a measure that is not a number where there were pairs to
  compute it from gives the worst score.
- **`AutoArima`.** The limits are filled one by one: `AutoArima{MaxP: 3}`
  keeps the defaults of the others (it used to set them all to zero, so the
  series was never differenced). A negative limit is a limit of zero.
- **`Quantile`** sorts values that are not numbers after all the others, as
  the crate does.
- Documentation: the errors of `Run`, what `AutoEts` leaves out, the length a
  decomposition needs, and a line for every exported name that had none.
