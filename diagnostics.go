package foresight

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
