package foresight

import (
	"math"
	"strconv"
)

// Regressors are named columns of values aligned with the ORIGINAL data: row
// i belongs to the observation whose position is i (see [Series.Index]). The
// rows go on past the end of the history, because a forecast needs the values
// of the regressors over the horizon.
//
// The zero value has no columns. The methods return a new value and leave
// the receiver as it was.
type Regressors struct {
	names   []string
	columns [][]float64
}

// With returns the regressors with one more column.
func (r Regressors) With(name string, values []float64) Regressors {
	return Regressors{
		names:   append(append([]string(nil), r.names...), name),
		columns: append(append([][]float64(nil), r.columns...), values),
	}
}

// And returns the regressors with the columns of another set added.
func (r Regressors) And(other Regressors) Regressors {
	return Regressors{
		names:   append(append([]string(nil), r.names...), other.names...),
		columns: append(append([][]float64(nil), r.columns...), other.columns...),
	}
}

// Fourier returns sines and cosines of the first order harmonics of a cycle
// of the given period, which need not be a whole number, for rows positions:
// a smooth seasonal pattern with few coefficients. The sine that is zero
// everywhere (the harmonic at half a whole period) is left out.
func Fourier(period float64, order, rows int) Regressors {
	var out Regressors
	label := strconv.FormatFloat(period, 'f', -1, 64)
	for k := 1; k <= order; k++ {
		sin, cos := make([]float64, rows), make([]float64, rows)
		for t := range rows {
			angle := 2 * math.Pi * float64(k) * float64(t) / period
			sin[t], cos[t] = math.Sin(angle), math.Cos(angle)
		}
		if math.Abs(2*float64(k)-period) > 1e-9 {
			out = out.With("sin"+strconv.Itoa(k)+"_"+label, sin)
		}
		out = out.With("cos"+strconv.Itoa(k)+"_"+label, cos)
	}
	return out
}

// SeasonalDummies returns one indicator for each position of the cycle but
// the first.
func SeasonalDummies(period, rows int) Regressors {
	var out Regressors
	for s := 1; s < period; s++ {
		column := make([]float64, rows)
		for t := range rows {
			if t%period == s {
				column[t] = 1
			}
		}
		out = out.With("season"+strconv.Itoa(s+1), column)
	}
	return out
}

// Names returns the names of the columns.
func (r Regressors) Names() []string { return r.names }

// Columns returns the columns. They are shared, not copied.
func (r Regressors) Columns() [][]float64 { return r.columns }

// Width returns the number of columns.
func (r Regressors) Width() int { return len(r.columns) }

// Rows returns the number of rows that every column has.
func (r Regressors) Rows() int {
	if len(r.columns) == 0 {
		return 0
	}
	rows := len(r.columns[0])
	for _, c := range r.columns {
		rows = min(rows, len(c))
	}
	return rows
}

// Covers reports whether there are rows for the positions 0 to end−1 and all
// of them are finite.
func (r Regressors) Covers(end int) bool {
	for _, c := range r.columns {
		if len(c) < end {
			return false
		}
		for _, v := range c[:end] {
			if !finite(v) {
				return false
			}
		}
	}
	return true
}
