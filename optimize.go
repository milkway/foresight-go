package foresight

import (
	"math"
	"slices"
	"sort"
)

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

// effort says how hard nelderMead tries.
type effort struct {
	// searches in a row, each from the best point of the one before
	runs int
	// relative spread of the values on the simplex at which a search stops
	tolerance float64
}

var (
	// enough to rank alternatives
	quick = effort{runs: 1, tolerance: 1e-8}
	// for the estimates that are reported
	thorough = effort{runs: 3, tolerance: 1e-11}
)

// below orders numbers with not-a-number after everything else, so that a
// point that cannot be evaluated is always the worst.
func below(a, b float64) bool {
	if math.IsNaN(a) {
		return false
	}
	if math.IsNaN(b) {
		return true
	}
	return a < b
}

type vertex struct {
	x     []float64
	value float64
}

// nelderMead is a simplex search from start, restarted from the best point
// with a smaller simplex while a restart still helps. f may return +Inf for
// points it cannot evaluate.
func nelderMead(f func([]float64) float64, start []float64, step float64, e effort) ([]float64, float64) {
	best := slices.Clone(start)
	value := f(best)
	if len(start) == 0 {
		return best, value
	}
	for range e.runs {
		x, v := simplex(f, best, value, step, e.tolerance)
		gain := value - v
		if v < value {
			best, value = x, v
		}
		if math.IsNaN(gain) || gain <= 1e-9*(1+math.Abs(value)) {
			break
		}
		step *= 0.5
	}
	return best, value
}

func simplex(f func([]float64) float64, start []float64, startValue, step, tolerance float64) ([]float64, float64) {
	n := len(start)
	points := make([]vertex, 0, n+1)
	points = append(points, vertex{slices.Clone(start), startValue})
	for i := range n {
		x := slices.Clone(start)
		x[i] += step
		points = append(points, vertex{x, f(x)})
	}
	order := func() {
		sort.SliceStable(points, func(a, b int) bool { return below(points[a].value, points[b].value) })
	}
	along := func(centre, worst []float64, t float64) []float64 {
		out := make([]float64, n)
		for i := range out {
			out[i] = centre[i] + t*(worst[i]-centre[i])
		}
		return out
	}
	for range 300 * n {
		order()
		low, high := points[0].value, points[n].value
		if math.Abs(high-low) <= tolerance*(math.Abs(low)+math.Abs(high))+1e-300 {
			break
		}
		centre := make([]float64, n)
		for _, p := range points[:n] {
			for i, v := range p.x {
				centre[i] += v / float64(n)
			}
		}
		worst := slices.Clone(points[n].x)
		second := points[n-1].value
		reflected := along(centre, worst, -1)
		fr := f(reflected)
		switch {
		case fr < low:
			expanded := along(centre, worst, -2)
			if fe := f(expanded); fe < fr {
				points[n] = vertex{expanded, fe}
			} else {
				points[n] = vertex{reflected, fr}
			}
		case fr < second:
			points[n] = vertex{reflected, fr}
		default:
			t := 0.5
			if fr < high {
				t = -0.5
			}
			contracted := along(centre, worst, t)
			if fc := f(contracted); fc < math.Min(fr, high) {
				points[n] = vertex{contracted, fc}
			} else {
				best := slices.Clone(points[0].x)
				for k := 1; k <= n; k++ {
					for i := range points[k].x {
						points[k].x[i] = best[i] + 0.5*(points[k].x[i]-best[i])
					}
					points[k].value = f(points[k].x)
				}
			}
		}
	}
	order()
	return points[0].x, points[0].value
}
