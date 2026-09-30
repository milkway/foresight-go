package foresight

import (
	"errors"
	"math"
)

// Series is a view over regularly spaced observations, oldest first.
//
// Besides the values, a series knows its seasonal period (12 for monthly data
// with a yearly cycle, 4 for quarterly, 1 for none) and where it sits in the
// original data: slicing keeps the absolute position, so the season of each
// observation and the alignment with external data (a price index, a
// regressor) survive any Head, Tail or Slice.
//
// The values are not copied: a Series shares them with the slice it was made
// from. The zero Series{} is an empty series without seasonality.
type Series struct {
	values []float64
	period int
	// season of the observation at absolute index 0
	phase int
	// absolute index of values[0]
	start int
	// set on a series handed to a worker of a parallel computation (or to a
	// backtest that is not to go parallel): whatever is computed from it, and
	// from its slices, stays on the calling goroutine
	alone bool
}

// NewSeries returns a series with the given seasonal period; a period under 1
// is read as 1 (no seasonality). The first observation is taken to be at
// season 0; see [Series.WithPhase].
func NewSeries(values []float64, period int) Series {
	return Series{values: values, period: max(period, 1)}
}

// NonSeasonal returns a series without seasonality.
func NonSeasonal(values []float64) Series {
	return NewSeries(values, 1)
}

// Monthly returns a series of monthly data with a yearly cycle; firstMonth is
// the calendar month of the first observation, 0 for January.
func Monthly(values []float64, firstMonth int) Series {
	return NewSeries(values, 12).WithPhase(firstMonth)
}

// Quarterly returns a series of quarterly data with a yearly cycle;
// firstQuarter is 0 for the first quarter.
func Quarterly(values []float64, firstQuarter int) Series {
	return NewSeries(values, 4).WithPhase(firstQuarter)
}

func mod(a, m int) int {
	return ((a % m) + m) % m
}

// WithPhase sets the season of the first observation of this view.
func (s Series) WithPhase(phase int) Series {
	s.phase = mod(phase-s.start, s.Period())
	return s
}

// Values returns the observations. The slice is shared, not copied.
func (s Series) Values() []float64 { return s.values }

// Len returns the number of observations.
func (s Series) Len() int { return len(s.values) }

// Period returns the seasonal period: 1 for a series without seasonality,
// the zero Series{} included.
func (s Series) Period() int { return max(s.period, 1) }

// Start returns the position, in the original data, of the first observation.
func (s Series) Start() int { return s.start }

// Index returns the position in the original data of observation i, which
// may be past the end: Len() is the first period to be forecast.
func (s Series) Index(i int) int { return s.start + i }

// Season returns the season (0 to Period()-1) of observation i, which, like
// in Index, may be past the end.
func (s Series) Season(i int) int {
	return mod(s.phase+s.start+i, s.Period())
}

// Head returns the first n observations (all of them if n is larger).
func (s Series) Head(n int) Series { return s.Slice(0, n) }

// Tail returns the last n observations (all of them if n is larger).
func (s Series) Tail(n int) Series {
	return s.Slice(max(s.Len()-n, 0), s.Len())
}

// Slice returns the observations from position from up to, but not
// including, position to, clamped to the series.
func (s Series) Slice(from, to int) Series {
	to = min(max(to, 0), s.Len())
	from = min(max(from, 0), to)
	s.values = s.values[from:to:to]
	s.start += from
	return s
}

// ErrLength is returned when two things that must have the same number of
// observations do not.
var ErrLength = errors.New("foresight: lengths differ")

// WithValues returns the same series (period, season and position) over
// other values, such as a transformation of the original ones.
func (s Series) WithValues(values []float64) (Series, error) {
	if len(values) != s.Len() {
		return Series{}, ErrLength
	}
	s.values = values
	return s, nil
}

// flat returns other values at the same position, without seasonality: what
// is left of this series once its seasonal patterns are taken out. Models
// that go by position (regressors, events, a deflator) stay aligned.
func (s Series) flat(values []float64) Series {
	return Series{values: values, period: 1, start: s.start, alone: s.alone}
}

// onOneGoroutine returns the series marked for computations that must not
// start goroutines of their own.
func (s Series) onOneGoroutine() Series {
	s.alone = true
	return s
}

func finite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// IsFinite reports whether every value is a finite number.
func (s Series) IsFinite() bool {
	for _, v := range s.values {
		if !finite(v) {
			return false
		}
	}
	return true
}

// IsPositive reports whether every value is finite and greater than zero,
// which is what multiplicative and logarithmic models need.
func (s Series) IsPositive() bool {
	for _, v := range s.values {
		if !finite(v) || v <= 0 {
			return false
		}
	}
	return true
}
