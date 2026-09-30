package foresight

import "errors"

// Errors returned by the models when a series does not suit them. A model
// never makes up a fit.
var (
	// ErrTooShort: the series has too few observations for the model.
	ErrTooShort = errors.New("foresight: series too short for the model")
	// ErrNotFinite: the series has values that are not finite numbers.
	ErrNotFinite = errors.New("foresight: series has values that are not finite")
	// ErrNotPositive: the model needs values greater than zero.
	ErrNotPositive = errors.New("foresight: model needs positive values")
	// ErrNotSeasonal: the model needs a seasonal period of at least 2.
	ErrNotSeasonal = errors.New("foresight: model needs a seasonal series")
	// ErrNoFit: the estimation did not reach a usable result.
	ErrNoFit = errors.New("foresight: model could not be fitted")
	// ErrConfig: the configuration of the model is not valid: a negative
	// order, a value that is not one of the named ones, a model that is
	// missing.
	ErrConfig = errors.New("foresight: invalid configuration of the model")
	// ErrHorizon: the horizon asked of [Forecast] is negative.
	ErrHorizon = errors.New("foresight: negative horizon")
	// ErrForecast: a forecast is not a finite number, as when regressors stop
	// short of the horizon or a transformation cannot be brought back.
	ErrForecast = errors.New("foresight: forecast is not finite")
)

// errPanic stands for a model that panicked where a panic cannot be let
// through: in a worker of a backtest or of an ensemble.
var errPanic = errors.New("foresight: model panicked")

// Param is a named number estimated by a model, for display and audit trails.
type Param struct {
	Name  string
	Value float64
}

// Model is a forecasting method, before seeing any data.
//
// Fitting is separate from forecasting so that a fitted model can be
// inspected and asked for any horizon without being estimated again. Models
// are plain configuration and safe to share between goroutines.
type Model interface {
	// Name is a short identifier, such as "holt_winters".
	Name() string
	// Description is a one-line description of the method.
	Description() string
	// Fit estimates the model. It returns an error when the series is too
	// short or otherwise unsuitable.
	//
	// A model that fits other models hands them the series it was given, a
	// slice of it or [Series.WithValues], rather than a new Series: that is
	// how an ensemble inside learns that it is being fitted in a worker of a
	// backtest and must not start goroutines of its own.
	Fit(y Series) (Fitted, error)
}

// Fitted is a model estimated on a series. It is plain data: it can be kept
// and asked for forecasts from several goroutines at once.
type Fitted interface {
	// Forecast returns point forecasts for the h periods after the last
	// observation, and nothing for an h of zero or less. They are not finite
	// where the model cannot say: past the rows of its regressors, for
	// instance. [Forecast] checks that.
	Forecast(h int) []float64
	// Params returns the estimated parameters.
	Params() []Param
}

// Forecast fits the model and forecasts the h periods after the last
// observation.
//
// Besides the errors of the model, it returns [ErrHorizon] for a negative h,
// [ErrConfig] for a nil model, [ErrRegressors] when the regressors of the
// model do not reach the horizon and [ErrForecast] when a forecast is not
// finite: regressors that stop short inside another model, or a
// transformation that cannot be brought back.
func Forecast(m Model, y Series, h int) ([]float64, error) {
	if m == nil {
		return nil, ErrConfig
	}
	if h < 0 {
		return nil, ErrHorizon
	}
	if c, ok := m.(interface{ covers(Series, int) error }); ok {
		if err := c.covers(y, h); err != nil {
			return nil, err
		}
	}
	fit, err := m.Fit(y)
	if err != nil {
		return nil, err
	}
	forecast := fit.Forecast(h)
	for _, v := range forecast {
		if !finite(v) {
			return nil, ErrForecast
		}
	}
	return forecast, nil
}

// tryForecast is [Forecast] for a model that may panic: the panic comes back
// as an error.
func tryForecast(m Model, y Series, h int) (forecast []float64, err error) {
	defer func() {
		if recover() != nil {
			forecast, err = nil, errPanic
		}
	}()
	return Forecast(m, y, h)
}

// tryFit fits a model that may panic and asks it for h forecasts and its
// parameters: the panic comes back as an error.
func tryFit(m Model, y Series, h int) (forecast []float64, params []Param, err error) {
	defer func() {
		if recover() != nil {
			forecast, params, err = nil, nil, errPanic
		}
	}()
	if m == nil {
		return nil, nil, ErrConfig
	}
	fit, err := m.Fit(y)
	if err != nil {
		return nil, nil, err
	}
	return fit.Forecast(h), fit.Params(), nil
}

// nameOf is the name of a model inside another, which may be missing.
func nameOf(m Model) string {
	if m == nil {
		return "invalid"
	}
	return m.Name()
}

// descriptionOf is the description of a model inside another, which may be
// missing.
func descriptionOf(m Model) string {
	if m == nil {
		return "No model"
	}
	return m.Description()
}
