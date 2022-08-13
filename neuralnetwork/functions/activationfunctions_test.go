package functions

import (
	"math"
	"testing"
)

func testValues(v1, v2 float64, dp int) bool {
	return math.Round(v1*math.Pow(10, float64(dp))) == math.Round(v2*math.Pow(10, float64(dp)))
}

func TestApplyRandom(t *testing.T) {
	t.Run("Testing random results", func(t *testing.T) {
		v := make([]float64, 10)
		for i := range v {
			v[i] = ApplyRandom(0)
		}
		ok := true
		for i, v1 := range v[:len(v)-2] {
			for _, v2 := range v[i+1:] {
				if v1 == v2 {
					ok = false
				}
			}
		}
		if !ok {
			t.Errorf("ApplyRandom() = failed to generate random numbers")
		}
	})
}

func Test_sigmoid(t *testing.T) {
	type args struct {
		v float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "-10",
			args: args{-10},
			want: 0.0000453978687024344,
		},
		{
			name: "-4",
			args: args{-4},
			want: 0.0179862099620916,
		},
		{
			name: "-1",
			args: args{-1},
			want: 0.268941421369995,
		},
		{
			name: "0",
			args: args{0},
			want: 0.5,
		},
		{
			name: "1.6",
			args: args{1.6},
			want: 0.832018385133924,
		},
		{
			name: "10",
			args: args{10},
			want: 0.999954602131298,
		},
		{
			name: "100",
			args: args{100},
			want: 1.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := sigmoid(tt.args.v); !testValues(got, tt.want, 5) {
				t.Errorf("sigmoid() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_dsigmoid(t *testing.T) {
	type args struct {
		v float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "-10",
			args: args{-10},
			want: -110,
		},
		{
			name: "-4",
			args: args{-4},
			want: -20,
		},
		{
			name: "-1",
			args: args{-1},
			want: -2,
		},
		{
			name: "0",
			args: args{0},
			want: 0,
		},
		{
			name: "1.6",
			args: args{1.6},
			want: -0.96,
		},
		{
			name: "10",
			args: args{10},
			want: -90,
		},
		{
			name: "100",
			args: args{100},
			want: -9900,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := dsigmoid(tt.args.v); !testValues(got, tt.want, 5) {
				t.Errorf("dsigmoid() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_relu(t *testing.T) {
	type args struct {
		v float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "-10",
			args: args{-10},
			want: 0,
		},
		{
			name: "-4",
			args: args{-4},
			want: 0,
		},
		{
			name: "-1",
			args: args{-1},
			want: 0,
		},
		{
			name: "0",
			args: args{0},
			want: 0,
		},
		{
			name: "1.6",
			args: args{1.6},
			want: 1.6,
		},
		{
			name: "10",
			args: args{10},
			want: 10,
		},
		{
			name: "100",
			args: args{100},
			want: 100,
		}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := relu(tt.args.v); !testValues(got, tt.want, 5) {
				t.Errorf("relu() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_drelu(t *testing.T) {
	type args struct {
		v float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "-10",
			args: args{-10},
			want: 0,
		},
		{
			name: "-4",
			args: args{-4},
			want: 0,
		},
		{
			name: "-1",
			args: args{-1},
			want: 0,
		},
		{
			name: "0",
			args: args{0},
			want: 0,
		},
		{
			name: "1.6",
			args: args{1.6},
			want: 1,
		},
		{
			name: "10",
			args: args{10},
			want: 1,
		},
		{
			name: "100",
			args: args{100},
			want: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := drelu(tt.args.v); !testValues(got, tt.want, 5) {
				t.Errorf("drelu() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_tanh(t *testing.T) {
	type args struct {
		v float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "-10",
			args: args{-10},
			want: -0.999999995877693,
		},
		{
			name: "-4",
			args: args{-4},
			want: -0.999329299739067,
		},
		{
			name: "-1",
			args: args{-1},
			want: -0.761594155955765,
		},
		{
			name: "0",
			args: args{0},
			want: 0,
		},
		{
			name: "1.6",
			args: args{1.6},
			want: 0.921668554406471,
		},
		{
			name: "10",
			args: args{10},
			want: 0.999999995877693,
		},
		{
			name: "100",
			args: args{100},
			want: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tanh(tt.args.v); !testValues(got, tt.want, 5) {
				t.Errorf("tanh() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_dtanh(t *testing.T) {
	type args struct {
		v float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "-10",
			args: args{-10},
			want: -99,
		},
		{
			name: "-4",
			args: args{-4},
			want: -15,
		},
		{
			name: "-1",
			args: args{-1},
			want: 0,
		},
		{
			name: "0",
			args: args{0},
			want: 1,
		},
		{
			name: "1.6",
			args: args{1.6},
			want: -1.56,
		},
		{
			name: "10",
			args: args{10},
			want: -99,
		},
		{
			name: "100",
			args: args{100},
			want: -9999,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := dtanh(tt.args.v); !testValues(got, tt.want, 5) {
				t.Errorf("dtanh() = %v, want %v", got, tt.want)
			}
		})
	}
}
