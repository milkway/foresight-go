package foresight

import (
	"math"
	"strconv"
)

// BoxCox is a transformation of the Box-Cox family: the logarithm for λ = 0,
// (y^λ − 1)/λ otherwise. It is defined for positive values.
type BoxCox struct {
	// Lambda is λ; zero is the logarithm.
	Lambda float64
}

// Apply transforms a value.
func (b BoxCox) Apply(y float64) float64 {
	if b.Lambda == 0 {
		return math.Log(y)
	}
	return (math.Pow(y, b.Lambda) - 1) / b.Lambda
}

// Invert brings a value back to the original scale. This is the median of
// the forecast distribution, not its mean.
func (b BoxCox) Invert(z float64) float64 {
	if b.Lambda == 0 {
		return math.Exp(z)
	}
	return math.Pow(math.Max(b.Lambda*z+1, 0), 1/b.Lambda)
}

// Guerrero returns the transformation with the λ in [−1, 2] that makes the
// variability most even across the cycles of the series (Guerrero, 1993): it
// minimises the coefficient of variation of sd / mean^(1 − λ) over blocks of
// one period (two observations when there is no seasonality). It reports
// false without at least two blocks of positive values.
func Guerrero(y Series) (BoxCox, bool) {
	if !y.IsPositive() {
		return BoxCox{}, false
	}
	size := max(y.Period(), 2)
	blocks := y.Len() / size
	if blocks < 2 {
		return BoxCox{}, false
	}
	v := y.Values()[y.Len()-blocks*size:]
	means, sds := make([]float64, blocks), make([]float64, blocks)
	for i := range blocks {
		b := v[i*size : (i+1)*size]
		means[i] = sum(b) / float64(size)
		s := 0.0
		for _, x := range b {
			s += (x - means[i]) * (x - means[i])
		}
		sds[i] = math.Sqrt(s / float64(size-1))
	}
	cv := func(lambda float64) float64 {
		r := make([]float64, blocks)
		for i := range r {
			r[i] = sds[i] / math.Pow(means[i], 1-lambda)
		}
		m := sum(r) / float64(blocks)
		s := 0.0
		for _, x := range r {
			s += (x - m) * (x - m)
		}
		value := math.Sqrt(s/float64(blocks-1)) / m
		if !finite(value) {
			return math.Inf(1)
		}
		return value
	}
	return BoxCox{golden(cv, -1, 2)}, true
}

// Transformed runs a model on Box-Cox transformed values and brings the
// forecasts back to the original scale. The series must be positive.
//
// Use [Log], [WithBoxCox] or [WithGuerrero] to make one.
type Transformed struct {
	// Model is the model of the transformed values.
	Model Model
	// Transform is the transformation, unless Automatic.
	Transform BoxCox
	// Automatic chooses λ by [Guerrero] at each fit.
	Automatic bool
}

// Log returns the model on the log scale.
func Log(m Model) Transformed { return Transformed{Model: m} }

// WithBoxCox returns the model on the Box-Cox scale with the given λ.
func WithBoxCox(m Model, lambda float64) Transformed {
	return Transformed{Model: m, Transform: BoxCox{lambda}}
}

// WithGuerrero returns the model on the Box-Cox scale with λ chosen by
// [Guerrero] at each fit.
func WithGuerrero(m Model) Transformed {
	return Transformed{Model: m, Automatic: true}
}

type transformedFit struct {
	inner     Fitted
	transform BoxCox
}

func (f transformedFit) Forecast(h int) []float64 {
	if h <= 0 {
		return nil
	}
	out := f.inner.Forecast(h)
	for i, z := range out {
		out[i] = f.transform.Invert(z)
	}
	return out
}

func (f transformedFit) Params() []Param {
	return append(f.inner.Params(), Param{"lambda", f.transform.Lambda})
}

func (t Transformed) isLog() bool { return !t.Automatic && t.Transform.Lambda == 0 }

// Name is the identifier of the model: "log_" or "boxcox_" and the name of
// the model inside ("log_invalid" without one).
func (t Transformed) Name() string {
	if t.isLog() {
		return "log_" + nameOf(t.Model)
	}
	return "boxcox_" + nameOf(t.Model)
}

// Description is a one-line description of the model.
func (t Transformed) Description() string {
	switch {
	case t.isLog():
		return descriptionOf(t.Model) + ", on the log scale"
	case t.Automatic:
		return descriptionOf(t.Model) + ", on the Box-Cox scale (λ by Guerrero's method)"
	}
	return descriptionOf(t.Model) + ", on the Box-Cox scale (λ = " +
		strconv.FormatFloat(t.Transform.Lambda, 'f', -1, 64) + ")"
}

// Fit transforms the series and fits the model inside. It returns
// [ErrConfig] without a model inside.
func (t Transformed) Fit(y Series) (Fitted, error) {
	if t.Model == nil {
		return nil, ErrConfig
	}
	if !y.IsPositive() {
		return nil, ErrNotPositive
	}
	transform := t.Transform
	if t.Automatic {
		var ok bool
		if transform, ok = Guerrero(y); !ok {
			return nil, ErrTooShort
		}
	}
	z := make([]float64, y.Len())
	for i, v := range y.Values() {
		z[i] = transform.Apply(v)
	}
	scaled, err := y.WithValues(z)
	if err != nil {
		return nil, err
	}
	inner, err := t.Model.Fit(scaled)
	if err != nil {
		return nil, err
	}
	return transformedFit{inner, transform}, nil
}

func (t Transformed) covers(y Series, h int) error {
	if c, ok := t.Model.(interface{ covers(Series, int) error }); ok {
		return c.covers(y, h)
	}
	return nil
}
