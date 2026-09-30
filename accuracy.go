package foresight

import (
	"cmp"
	"math"
	"slices"
)

// totalOrder compares two numbers in the total order of IEEE 754: −NaN, −Inf,
// …, −0, +0, …, +Inf, NaN. Values that are not numbers end up last (first
// with the sign bit set) instead of upsetting the sort.
func totalOrder(a, b float64) int {
	key := func(x float64) int64 {
		bits := int64(math.Float64bits(x))
		return bits ^ int64(uint64(bits>>63)>>1)
	}
	return cmp.Compare(key(a), key(b))
}

// Quantile returns the quantile p of v with linear interpolation (type 7,
// the default in R); p under 0 is read as 0 and over 1 as 1. v does not need
// to be sorted; values that are not numbers are sorted after all the others,
// so the quantiles that reach them are not numbers either. It reports false
// for an empty slice or a p that is not a number.
func Quantile(v []float64, p float64) (float64, bool) {
	if len(v) == 0 || math.IsNaN(p) {
		return 0, false
	}
	s := slices.Clone(v)
	slices.SortFunc(s, totalOrder)
	pos := min(max(p, 0), 1) * float64(len(s)-1)
	i := int(math.Floor(pos))
	f := pos - math.Floor(pos)
	if i+1 < len(s) {
		return s[i] + (s[i+1]-s[i])*f, true
	}
	return s[i], true
}

// mean of the values; NaN for none.
func mean(v []float64) float64 {
	if len(v) == 0 {
		return math.NaN()
	}
	return sum(v) / float64(len(v))
}

// MAPE is the mean absolute percentage error, in percent. Pairs whose actual
// value is zero are left out. NaN when there is no pair.
func MAPE(actual, forecast []float64) float64 {
	var e []float64
	for i := range min(len(actual), len(forecast)) {
		if actual[i] != 0 {
			e = append(e, math.Abs(forecast[i]-actual[i])/math.Abs(actual[i]))
		}
	}
	return mean(e) * 100
}

// Bias is the mean of (forecast − actual) / actual, in percent: positive
// when the forecasts run above what happened. Pairs whose actual value is
// zero are left out.
func Bias(actual, forecast []float64) float64 {
	var e []float64
	for i := range min(len(actual), len(forecast)) {
		if actual[i] != 0 {
			e = append(e, (forecast[i]-actual[i])/actual[i])
		}
	}
	return mean(e) * 100
}

// MAE is the mean absolute error.
func MAE(actual, forecast []float64) float64 {
	var e []float64
	for i := range min(len(actual), len(forecast)) {
		e = append(e, math.Abs(forecast[i]-actual[i]))
	}
	return mean(e)
}

// RMSE is the root mean squared error.
func RMSE(actual, forecast []float64) float64 {
	var e []float64
	for i := range min(len(actual), len(forecast)) {
		d := forecast[i] - actual[i]
		e = append(e, d*d)
	}
	return math.Sqrt(mean(e))
}

// MASEScale is the scale of the mean absolute scaled error: the in-sample
// mean absolute error of the seasonal naive forecast (naive when period is
// 1). It reports false when the training data is no longer than one period
// or the scale is zero.
func MASEScale(train []float64, period int) (float64, bool) {
	m := max(period, 1)
	if len(train) <= m {
		return 0, false
	}
	s := 0.0
	for t := m; t < len(train); t++ {
		s += math.Abs(train[t] - train[t-m])
	}
	s /= float64(len(train) - m)
	return s, s > 0
}

// MASE is the mean absolute scaled error given the scale from [MASEScale].
func MASE(actual, forecast []float64, scale float64) float64 {
	return MAE(actual, forecast) / scale
}
