package foresight

import "math"

// golden returns the minimum of a function of one variable on [lo, hi] by
// golden section.
func golden(f func(float64) float64, lo, hi float64) float64 {
	ratio := (math.Sqrt(5) - 1) / 2
	a, b := hi-ratio*(hi-lo), lo+ratio*(hi-lo)
	fa, fb := f(a), f(b)
	for range 80 {
		if fa < fb {
			hi = b
			b, fb = a, fa
			a = hi - ratio*(hi-lo)
			fa = f(a)
		} else {
			lo = a
			a, fa = b, fb
			b = lo + ratio*(hi-lo)
			fb = f(b)
		}
	}
	return (lo + hi) / 2
}
