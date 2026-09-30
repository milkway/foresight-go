package foresight

import (
	"math"
	"slices"
	"sort"
	"strings"
)

// Weighting says how the forecasts of the members of an ensemble are
// combined.
type Weighting int

const (
	// InverseError weights in inverse proportion to the mean squared error
	// each member had when forecasting the end of the history (the default).
	InverseError Weighting = iota
	// EqualWeights is the plain average.
	EqualWeights
	// Median is the median, which a member gone astray cannot drag along.
	Median
	// Stacked uses the weights, none negative and adding up to one, that
	// would have given the combination with the smallest squared error over
	// the end of the history. Members that add nothing get exactly zero.
	Stacked
)

// Ensemble is a model made of other models.
//
// The weights are learnt from the series itself: the last Origins periods
// are forecast by every member from the data before them, as in a backtest,
// and the errors decide how much each member counts. Then the members are
// fitted on the whole series and their forecasts combined. Members that
// cannot be fitted are left out.
//
// Being a model, an ensemble can be a candidate in a [Backtest], next to its
// own members: its weights are then learnt again at every origin, from what
// was known at that point.
type Ensemble struct {
	Members   []Candidate
	Weighting Weighting
	// Origins is how many of the last periods are forecast to learn the
	// weights (default 12).
	Origins int
	// Horizon is how far ahead each of those forecasts goes (default: one
	// seasonal cycle, at most 12 periods).
	Horizon int
	// Top keeps only the members with the smallest error; zero keeps all.
	Top int
}

// errors returns, for each member, the errors (forecast − actual, in units
// of the mean size of the series) over the end of the history; nil for a
// member that could not forecast from every origin, and for all of them when
// the series is too short to set the end apart.
func (e Ensemble) errors(y Series) [][]float64 {
	v, n := y.Values(), y.Len()
	out := make([][]float64, len(e.Members))
	horizon := e.Horizon
	if horizon <= 0 {
		horizon = min(max(y.Period(), 1), 12)
	}
	origins := e.Origins
	if origins <= 0 {
		origins = 12
	}
	origins = min(origins, max(n-max(2*y.Period(), 8), 0))
	if origins == 0 {
		return out
	}
	first := n - origins
	scale := 0.0
	for _, x := range v {
		scale += abs(x)
	}
	scale /= float64(n)
	if scale <= 0 {
		scale = 1
	}
	for i, member := range e.Members {
		forecasts := mapIndices(origins, false, func(k int) []float64 {
			p, err := Forecast(member.Model, y.Head(first+k), horizon)
			if err != nil || len(p) != horizon {
				return nil
			}
			for _, x := range p {
				if !finite(x) {
					return nil
				}
			}
			return p
		})
		if slices.ContainsFunc(forecasts, func(p []float64) bool { return p == nil }) {
			continue
		}
		errs := []float64{}
		for k, forecast := range forecasts {
			for h, f := range forecast {
				if first+k+h < n {
					errs = append(errs, (f-v[first+k+h])/scale)
				}
			}
		}
		out[i] = errs
	}
	return out
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// ontoSimplex returns the point of the simplex (no negative coordinate, sum
// one) closest to v.
func ontoSimplex(v []float64) []float64 {
	sorted := slices.Clone(v)
	sort.Sort(sort.Reverse(sort.Float64Slice(sorted)))
	running, shift := 0.0, 0.0
	for i, x := range sorted {
		running += x
		if candidate := (running - 1) / float64(i+1); x-candidate > 0 {
			shift = candidate
		}
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = max(x-shift, 0)
	}
	return out
}

// stackedWeights returns the weights on the simplex that minimise the
// squared length of the weighted sum of the error vectors, by projected
// gradient with momentum.
func stackedWeights(errors [][]float64) []float64 {
	m := len(errors)
	gram := make([][]float64, m)
	steepest := 0.0
	for i := range gram {
		gram[i] = make([]float64, m)
		for j := range gram[i] {
			gram[i][j] = dot(errors[i], errors[j])
		}
		steepest += gram[i][i]
	}
	steepest *= 2
	w := repeat(1/float64(m), m)
	if steepest <= 0 {
		return w
	}
	value := func(w []float64) float64 {
		s := 0.0
		for i := range w {
			s += w[i] * dot(gram[i], w)
		}
		return s
	}
	ahead := slices.Clone(w)
	momentum := 1.0
	for range 20_000 {
		step := make([]float64, m)
		for i := range step {
			step[i] = ahead[i] - 2*dot(gram[i], ahead)/steepest
		}
		next := ontoSimplex(step)
		moved := 0.0
		for i := range next {
			moved += abs(next[i] - w[i])
		}
		// momentum that overshoots is dropped
		if value(next) > value(w) {
			ahead = slices.Clone(w)
			momentum = 1
			continue
		}
		faster := (1 + math.Sqrt(1+4*momentum*momentum)) / 2
		for i := range ahead {
			ahead[i] = next[i] + (momentum-1)/faster*(next[i]-w[i])
		}
		w = next
		if moved < 1e-13 {
			break
		}
		momentum = faster
	}
	// what is numerically nothing is nothing
	total := 0.0
	for i, x := range w {
		if x < 1e-9 {
			w[i] = 0
		}
		total += w[i]
	}
	for i := range w {
		w[i] /= total
	}
	return w
}

type ensembleMember struct {
	name   string
	weight float64
	fit    Fitted
	errors []float64
}

type ensembleFit struct {
	members []ensembleMember
	median  bool
}

func (f ensembleFit) Forecast(h int) []float64 {
	each := make([][]float64, len(f.members))
	for i, m := range f.members {
		each[i] = m.fit.Forecast(h)
	}
	out := make([]float64, h)
	for k := range out {
		if f.median {
			column := make([]float64, len(each))
			for i := range each {
				column[i] = each[i][k]
			}
			out[k], _ = Quantile(column, 0.5)
			continue
		}
		for i, m := range f.members {
			out[k] += m.weight * each[i][k]
		}
	}
	return out
}

func (f ensembleFit) Params() []Param {
	out := make([]Param, len(f.members))
	for i, m := range f.members {
		out[i] = Param{"weight_" + m.name, m.weight}
	}
	return out
}

func (e Ensemble) Name() string {
	return "ensemble_" + [...]string{"inverse_error", "equal", "median", "stacked"}[e.Weighting]
}

func (e Ensemble) Description() string {
	how := [...]string{
		"average weighted by inverse error", "average", "median",
		"combination with the weights of least error",
	}[e.Weighting]
	names := make([]string, len(e.Members))
	for i, m := range e.Members {
		names[i] = m.Name
	}
	return "Ensemble, " + how + ", of " + strings.Join(names, ", ")
}

func meanSquare(e []float64) float64 { return dot(e, e) / float64(len(e)) }

func (e Ensemble) Fit(y Series) (Fitted, error) {
	learnt := e.Weighting == InverseError || e.Weighting == Stacked || e.Top > 0
	errs := make([][]float64, len(e.Members))
	if learnt {
		errs = e.errors(y)
	}
	// members fitted on the whole series, with their errors if any
	var fitted []ensembleMember
	for i, m := range e.Members {
		if fit, err := m.Model.Fit(y); err == nil {
			fitted = append(fitted, ensembleMember{name: m.Name, fit: fit, errors: errs[i]})
		}
	}
	if len(fitted) == 0 {
		return nil, ErrNoCandidate
	}
	has := func(m ensembleMember) bool { return len(m.errors) > 0 }
	// errors can only decide if everyone left has them
	if learnt && slices.ContainsFunc(fitted, func(m ensembleMember) bool { return m.errors != nil }) {
		fitted = slices.DeleteFunc(fitted, func(m ensembleMember) bool { return !has(m) })
	}
	measured := !slices.ContainsFunc(fitted, func(m ensembleMember) bool { return m.errors == nil })
	if e.Top > 0 && measured {
		sort.SliceStable(fitted, func(a, b int) bool {
			return meanSquare(fitted[a].errors) < meanSquare(fitted[b].errors)
		})
		fitted = fitted[:min(e.Top, len(fitted))]
	}
	m := len(fitted)
	weights := repeat(1/float64(m), m)
	switch {
	case e.Weighting == InverseError && measured:
		scores := make([]float64, m)
		perfect, total := 0, 0.0
		for i, f := range fitted {
			scores[i] = meanSquare(f.errors)
			if scores[i] == 0 {
				perfect++
			} else {
				total += 1 / scores[i]
			}
		}
		for i, s := range scores {
			switch {
			case perfect > 0 && s == 0:
				// whoever made no error takes it all
				weights[i] = 1 / float64(perfect)
			case perfect > 0:
				weights[i] = 0
			default:
				weights[i] = 1 / s / total
			}
		}
	case e.Weighting == Stacked && measured:
		vectors := make([][]float64, m)
		for i, f := range fitted {
			vectors[i] = f.errors
		}
		weights = stackedWeights(vectors)
	}
	for i := range fitted {
		fitted[i].weight = weights[i]
	}
	return ensembleFit{fitted, e.Weighting == Median}, nil
}
