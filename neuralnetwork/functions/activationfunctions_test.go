package functions

import (
	"fmt"
	"math"
	"testing"
)

func TestApplyRandom(t *testing.T) {
	type args struct {
		v float64
	}
	tests := make([]struct {
		name string
		args args
	}, 10)
	for i, t := range tests {
		t.name = fmt.Sprintf("Test %d", i)
		t.args = args{float64(i)}
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ApplyRandom(tt.args.v); got < 0 {
				t.Errorf("ApplyRandom() = %v, want > 0", got)
			}
		})
	}
}

func TestSigmoid(t *testing.T) {
	type args struct {
		v float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "Sigmoid 3.6",
			args: args{3.6},
			want: 0.973,
		},
		{
			name: "Sigmoid 0.8",
			args: args{0.8},
			want: 0.690,
		},
		{
			name: "Sigmoid 3.1",
			args: args{3.1},
			want: 0.957,
		},
		{
			name: "Sigmoid 3.9",
			args: args{3.9},
			want: 0.980,
		},
		{
			name: "Sigmoid 0",
			args: args{0},
			want: 0.5,
		},
		{
			name: "Sigmoid 1.5",
			args: args{1.5},
			want: 0.818,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := math.Round(Sigmoid(tt.args.v)*1000) / 1000; got != tt.want {
				t.Errorf("Sigmoid() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDsigmoid(t *testing.T) {
	type args struct {
		v float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "Dsigmoid 2",
			args: args{2},
			want: -2,
		},
		{
			name: "Dsigmoid 0.7",
			args: args{0.7},
			want: 0.21,
		},
		{
			name: "Dsigmoid 0.1",
			args: args{0.1},
			want: 0.09,
		},
		{
			name: "Dsigmoid 3.9",
			args: args{3.9},
			want: -11.31,
		},
		{
			name: "Dsigmoid 0",
			args: args{0},
			want: 0,
		},
		{
			name: "Dsigmoid 1.8",
			args: args{1.8},
			want: -1.44,
		},
	}
	for i, t := range tests {
		v := float64(i)
		t.name = fmt.Sprintf("Test %d", i)
		t.args = args{float64(v)}
		t.want = v * (1 - v)
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := math.Round(Dsigmoid(tt.args.v)*1000) / 1000; got != tt.want {
				t.Errorf("Dsigmoid() = %v, want %v", got, tt.want)
			}
		})
	}
}
