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
)

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
	Fit(y Series) (Fitted, error)
}

// Fitted is a model estimated on a series.
type Fitted interface {
	// Forecast returns point forecasts for the h periods after the last
	// observation.
	Forecast(h int) []float64
	// Params returns the estimated parameters.
	Params() []Param
}

// Forecast fits the model and forecasts the h periods after the last
// observation. A model with regressors that do not reach the horizon gives
// an error rather than a forecast that is not a number.
func Forecast(m Model, y Series, h int) ([]float64, error) {
	if c, ok := m.(interface{ covers(Series, int) error }); ok {
		if err := c.covers(y, h); err != nil {
			return nil, err
		}
	}
	fit, err := m.Fit(y)
	if err != nil {
		return nil, err
	}
	return fit.Forecast(h), nil
}
