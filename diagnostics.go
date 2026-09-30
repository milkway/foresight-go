package foresight

import "math"

// ACF returns the autocorrelations of y at lags 1 to maxLag.
func ACF(y []float64, maxLag int) []float64 {
	n := len(y)
	m := sum(y) / float64(n)
	c0 := 0.0
	for _, v := range y {
		c0 += (v - m) * (v - m)
	}
	out := make([]float64, maxLag)
	for k := 1; k <= maxLag; k++ {
		ck := 0.0
		for t := k; t < n; t++ {
			ck += (y[t] - m) * (y[t-k] - m)
		}
		out[k-1] = ck / c0
	}
	return out
}

// centredAverage is the moving average of one full cycle centred on t.
func centredAverage(y []float64, t, m int) float64 {
	half := m / 2
	if m%2 == 0 {
		inner := sum(y[t+1-half : t+half])
		return (0.5*y[t-half] + inner + 0.5*y[t+half]) / float64(m)
	}
	return sum(y[t-half:t+half+1]) / float64(m)
}

// Difference returns y differenced once at the given lag (1 for the ordinary
// difference).
func Difference(y []float64, lag int) []float64 {
	if lag >= len(y) {
		return nil
	}
	out := make([]float64, len(y)-lag)
	for t := lag; t < len(y); t++ {
		out[t-lag] = y[t] - y[t-lag]
	}
	return out
}

func variance(v []float64) (float64, bool) {
	n := len(v)
	if n < 2 {
		return 0, false
	}
	m := sum(v) / float64(n)
	s := 0.0
	for _, x := range v {
		s += (x - m) * (x - m)
	}
	return s / float64(n-1), true
}

// KPSS5Percent is the 5% critical value of the KPSS test for level
// stationarity.
const KPSS5Percent = 0.463

// KPSS returns the statistic of the test of Kwiatkowski, Phillips, Schmidt &
// Shin (1992) for the null hypothesis that the series is stationary around a
// constant level, with a Bartlett window of ⌊3√n / 13⌋ lags. Large values
// reject stationarity; the 5% critical value is [KPSS5Percent]. It reports
// false for fewer than 3 observations or a statistic that is not finite.
func KPSS(y []float64) (float64, bool) {
	n := len(y)
	if n < 3 {
		return 0, false
	}
	nf := float64(n)
	m := sum(y) / nf
	e := make([]float64, n)
	cumulative, eta := 0.0, 0.0
	for i, v := range y {
		e[i] = v - m
		cumulative += e[i]
		eta += cumulative * cumulative
	}
	eta /= nf * nf
	lags := int(math.Floor(3 * math.Sqrt(nf) / 13))
	longRun := 0.0
	for _, v := range e {
		longRun += v * v
	}
	longRun /= nf
	for j := 1; j <= min(lags, n-1); j++ {
		weight := 1 - float64(j)/float64(lags+1)
		cross := 0.0
		for t := j; t < n; t++ {
			cross += e[t] * e[t-j]
		}
		longRun += 2 * weight * cross / nf
	}
	stat := eta / longRun
	return stat, finite(stat)
}

// NDiffs returns the number of ordinary differences needed for the series to
// pass the KPSS test at 5%, at most limit.
func NDiffs(y []float64, limit int) int {
	current := y
	d := 0
	for d < limit {
		stat, ok := KPSS(current)
		if !ok || stat <= KPSS5Percent {
			break
		}
		current = Difference(current, 1)
		d++
	}
	return d
}

// SeasonalStrengthThreshold is the seasonal strength above which a seasonal
// difference is taken.
const SeasonalStrengthThreshold = 0.64

// SeasonalStrength returns the strength of seasonality in [0, 1] (Wang, Smith
// & Hyndman, 2006): 1 − Var(remainder) / Var(seasonal + remainder), from a
// classical additive decomposition (centred moving average and seasonal
// means). It reports false for non-seasonal or short series (fewer than two
// cycles plus one observation).
func SeasonalStrength(y []float64, period int) (float64, bool) {
	n, m := len(y), period
	if m < 2 || n < 2*m+1 {
		return 0, false
	}
	half := m / 2
	type point struct {
		season int
		value  float64
	}
	var detrended []point
	for t := half; t < n-half; t++ {
		detrended = append(detrended, point{t % m, y[t] - centredAverage(y, t, m)})
	}
	total := make([]float64, m)
	count := make([]int, m)
	for _, p := range detrended {
		total[p.season] += p.value
		count[p.season]++
	}
	seasonal := make([]float64, m)
	for i := range seasonal {
		if count[i] == 0 {
			return 0, false
		}
		seasonal[i] = total[i] / float64(count[i])
	}
	centre := sum(seasonal) / float64(m)
	for i := range seasonal {
		seasonal[i] -= centre
	}
	remainder := make([]float64, len(detrended))
	both := make([]float64, len(detrended))
	for i, p := range detrended {
		remainder[i] = p.value - seasonal[p.season]
		both[i] = p.value
	}
	vr, ok1 := variance(remainder)
	vb, ok2 := variance(both)
	if !ok1 || !ok2 {
		return 0, false
	}
	if vb <= 0 {
		return 0, true
	}
	return min(max(1-vr/vb, 0), 1), true
}

// NSDiffs returns the number of seasonal differences (0 or 1) suggested by
// the strength of the seasonality.
func NSDiffs(y []float64, period int) int {
	if s, ok := SeasonalStrength(y, period); ok && s > SeasonalStrengthThreshold {
		return 1
	}
	return 0
}
