package foresight

import "math"

var inf = math.Inf(1)

// The rest of the package takes series without gaps. In the functions below
// a gap is a value that is not a number (math.NaN()), and what they return is
// ready to be modelled.

func linearFill(values []float64) ([]float64, bool) {
	var known []int
	for i, v := range values {
		if finite(v) {
			known = append(known, i)
		}
	}
	if len(known) == 0 {
		return nil, false
	}
	first, last := known[0], known[len(known)-1]
	out := append([]float64(nil), values...)
	// the ends take the nearest value
	for i := range first {
		out[i] = values[first]
	}
	for i := last + 1; i < len(out); i++ {
		out[i] = values[last]
	}
	for k := 1; k < len(known); k++ {
		a, b := known[k-1], known[k]
		for i := a + 1; i < b; i++ {
			share := float64(i-a) / float64(b-a)
			out[i] = values[a] + share*(values[b]-values[a])
		}
	}
	return out, true
}

// Interpolate fills the gaps. Without seasonality (period under 2, or a
// series not longer than two full cycles) by straight lines between the
// neighbours; with it, the straight lines are drawn on the seasonally
// adjusted series and the seasonal pattern is put back, so a gap in December
// is filled with a December. It reports false when no value is known.
func Interpolate(values []float64, period int) ([]float64, bool) {
	first, ok := linearFill(values)
	if !ok {
		return nil, false
	}
	gaps := false
	for _, v := range values {
		if !finite(v) {
			gaps = true
			break
		}
	}
	if !gaps {
		return first, true
	}
	parts, err := Stl{Period: period, Robust: true}.Decompose(first)
	if err != nil {
		return first, true
	}
	seasonal := parts.Seasonal[0]
	adjusted := make([]float64, len(values))
	for i, v := range values {
		adjusted[i] = v - seasonal[i]
	}
	filled, ok := linearFill(adjusted)
	if !ok {
		return nil, false
	}
	for i := range filled {
		filled[i] += seasonal[i]
	}
	return filled, true
}

// Outlier is an observation that does not fit with the others.
type Outlier struct {
	// Index is the position in the series.
	Index int
	// Value is what was observed.
	Value float64
	// Replacement is what the neighbours and the season suggest instead.
	Replacement float64
}

// robustStrength is the strength of the seasonal pattern measured with
// interquartile ranges in place of variances, so that the outliers being
// looked for do not pass for a weak pattern.
func robustStrength(seasonal, remainder []float64) float64 {
	spread := func(v []float64) float64 {
		q1, ok1 := Quantile(v, 0.25)
		q3, ok2 := Quantile(v, 0.75)
		if !ok1 || !ok2 {
			return 0
		}
		return (q3 - q1) * (q3 - q1)
	}
	both := make([]float64, len(seasonal))
	for i := range both {
		both[i] = seasonal[i] + remainder[i]
	}
	total := spread(both)
	if total <= 0 {
		return 0
	}
	return min(max(1-spread(remainder)/total, 0), 1)
}

// Outliers finds the observations whose distance from trend and seasonality
// is more than three interquartile ranges beyond the quartiles of those
// distances.
//
// Trend and seasonality come from a robust STL decomposition; the seasonal
// pattern is only taken out when it is strong (strength of at least 0.6,
// measured with interquartile ranges), as a weak one would be mostly noise.
// Series without seasonality get a robust local linear trend. Gaps are
// filled before looking. It reports false when the series has fewer than 8
// known values.
func Outliers(values []float64, period int) ([]Outlier, bool) {
	known := 0
	for _, v := range values {
		if finite(v) {
			known++
		}
	}
	if known < 8 {
		return nil, false
	}
	filled, ok := Interpolate(values, period)
	if !ok {
		return nil, false
	}
	n := len(values)
	smooth := make([]float64, n)
	if parts, err := (Stl{Period: period, SeasonalWindow: 11, Robust: true}).Decompose(filled); err == nil {
		copy(smooth, parts.Trend)
		if robustStrength(parts.Seasonal[0], parts.Remainder) >= 0.6 {
			for i := range smooth {
				smooth[i] += parts.Seasonal[0][i]
			}
		}
	} else {
		// a robust trend for a series without seasonality: the trend of a
		// decomposition whose "cycle" of two periods carries no pattern to
		// speak of
		parts, err := Stl{Period: 2, TrendWindow: max(n/5, 7), Robust: true}.Decompose(filled)
		if err != nil {
			return nil, false
		}
		copy(smooth, parts.Trend)
	}
	distance := make([]float64, n)
	for i := range distance {
		distance[i] = filled[i] - smooth[i]
	}
	q1, _ := Quantile(distance, 0.25)
	q3, _ := Quantile(distance, 0.75)
	// on exact data the spread is rounding noise: nothing is that far from it
	size := 0.0
	for _, v := range filled {
		size = max(size, math.Abs(v))
	}
	reach := max(3*(q3-q1), 1e-9*size)
	var flagged []int
	for i, v := range values {
		if finite(v) && (distance[i] < q1-reach || distance[i] > q3+reach) {
			flagged = append(flagged, i)
		}
	}
	if len(flagged) == 0 {
		return []Outlier{}, true
	}
	holes := append([]float64(nil), values...)
	for _, i := range flagged {
		holes[i] = math.NaN()
	}
	replaced, ok := Interpolate(holes, period)
	if !ok {
		return nil, false
	}
	out := make([]Outlier, len(flagged))
	for k, i := range flagged {
		out[k] = Outlier{i, values[i], replaced[i]}
	}
	return out, true
}

// Clean returns the series with the gaps filled and the outliers replaced.
// It reports false when fewer than 8 values are known.
func Clean(values []float64, period int) ([]float64, bool) {
	out, ok := Interpolate(values, period)
	if !ok {
		return nil, false
	}
	found, ok := Outliers(values, period)
	if !ok {
		return nil, false
	}
	for _, o := range found {
		out[o.Index] = o.Replacement
	}
	return out, true
}
