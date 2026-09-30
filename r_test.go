package foresight

import (
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
)

// column reads a column of a file of testdata: i = 1 for the first column
// after the month.
func column(t testing.TB, file string, i int) []float64 {
	t.Helper()
	raw, err := os.ReadFile("testdata/" + file)
	if err != nil {
		t.Fatal(err)
	}
	var out []float64
	for _, l := range strings.Split(string(raw), "\n") {
		if l == "" || strings.HasPrefix(l, "#") || strings.HasPrefix(l, "month") {
			continue
		}
		v, err := strconv.ParseFloat(strings.Split(l, ",")[i], 64)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, v)
	}
	return out
}

func icms(t testing.TB) []float64 { return column(t, "piaui_revenue.csv", 1) }
func fpe(t testing.TB) []float64  { return column(t, "piaui_revenue.csv", 2) }
func ipca(t testing.TB) []float64 { return column(t, "piaui_revenue.csv", 3) }
func air(t testing.TB) []float64  { return column(t, "air_passengers.csv", 1) }

func closeTo(t *testing.T, name string, ours, reference []float64, tolerance float64) {
	t.Helper()
	if len(ours) != len(reference) {
		t.Fatalf("%s: %d values, want %d", name, len(ours), len(reference))
	}
	for h := range ours {
		if math.Abs(ours[h]/reference[h]-1) >= tolerance {
			t.Errorf("%s h=%d: %v, want %v", name, h+1, ours[h], reference[h])
		}
	}
}

// Forecasts of the R package forecast 9.0.2 on public data; the reference
// values come from the scripts in tests/r of the Rust crate.
type reference struct {
	theta, snaive, drift []float64
}

var rICMS = reference{
	theta: []float64{
		821180569.452, 826589211.539, 825297914.497, 854821247.833,
		842379893.905, 875039597.321, 875547670.792, 758722396.695,
		705075455.027, 771713771.372, 729860110.323, 812225243.518,
	},
	snaive: []float64{
		725046345.29, 773297361.32, 747978074.80, 751243471.44,
		771989639.97, 767244468.11, 791797624.93, 720099765.32,
		679758370.35, 774001802.81, 704801950.30, 785725856.19,
	},
	drift: []float64{
		790441292.432, 795156728.674, 799872164.916, 804587601.158,
		809303037.399, 814018473.641, 818733909.883, 823449346.125,
		828164782.367, 832880218.609, 837595654.851, 842311091.093,
	},
}

var rFPE = reference{
	theta: []float64{
		673005978.597, 807295251.993, 658367288.813, 717159016.694,
		905840242.167, 1021426816.045, 964076441.901, 1267218459.982,
		813672292.609, 853565729.946, 997969347.093, 911315990.981,
	},
	snaive: []float64{
		521384984.85, 685235336.58, 553829994.03, 585488783.66,
		790424639.46, 892435659.23, 829385876.87, 1044022025.38,
		631613479.02, 744050636.75, 900471019.35, 949640302.02,
	},
	drift: []float64{
		955919073.134, 962197844.247, 968476615.361, 974755386.475,
		981034157.588, 987312928.702, 993591699.816, 999870470.930,
		1006149242.043, 1012428013.157, 1018706784.271, 1024985555.384,
	},
}

func TestBenchmarksMatchR(t *testing.T) {
	for _, c := range []struct {
		name   string
		values []float64
		r      reference
	}{{"icms", icms(t), rICMS}, {"fpe", fpe(t), rFPE}} {
		y := Monthly(c.values, 2)
		p, err := Forecast(SeasonalNaive{}, y, 12)
		if err != nil {
			t.Fatal(err)
		}
		closeTo(t, c.name+" snaive", p, c.r.snaive, 1e-11)
		p, err = Forecast(Drift{}, y, 12)
		if err != nil {
			t.Fatal(err)
		}
		closeTo(t, c.name+" drift", p, c.r.drift, 1e-11)
	}
}

// R estimates the smoothing parameter and the initial level with a general
// optimiser and we use a closed form plus a line search, so the optimum is
// the same only up to the optimiser's precision.
func TestThetaMatchesR(t *testing.T) {
	for _, c := range []struct {
		name   string
		values []float64
		r      reference
	}{{"icms", icms(t), rICMS}, {"fpe", fpe(t), rFPE}} {
		p, err := Forecast(Theta{}, Monthly(c.values, 2), 12)
		if err != nil {
			t.Fatal(err)
		}
		closeTo(t, c.name+" theta", p, c.r.theta, 1e-3)
	}
}

func TestDataIsWhatItShouldBe(t *testing.T) {
	if len(icms(t)) != 112 || len(fpe(t)) != 112 || len(ipca(t)) != 112 || len(air(t)) != 144 {
		t.Error("length of the series")
	}
}
