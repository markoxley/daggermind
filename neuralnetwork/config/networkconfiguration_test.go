package config

import (
	"reflect"
	"testing"

	"github.com/markoxley/daggermind/neuralnetwork/functions"
)

func TestNew(t *testing.T) {
	type args struct {
		topology []uint32
	}
	tests := []struct {
		name string
		args args
		want *NetworkConfiguration
	}{
		{name: "standard - 2 inputs, 2 outputs, 4 hidden",
			args: args{[]uint32{2, 4, 2}},
			want: &NetworkConfiguration{
				Topology:     []uint32{2, 4, 2},
				LearningRate: 0.1,
				Functions:    functions.Sigmoid,
				Quiet:        true,
			},
		},
		{name: "deep - 4 inputs, 3 outputs, 2 x 4 hidden",
			args: args{[]uint32{2, 4, 4, 2}},
			want: &NetworkConfiguration{
				Topology:     []uint32{2, 4, 4, 2},
				LearningRate: 0.1,
				Functions:    functions.Sigmoid,
				Quiet:        true,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := New(tt.args.topology); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("New() = %v, want %v", got, tt.want)
			}
		})
	}
}
