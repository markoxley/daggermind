package network

import (
	"reflect"
	"testing"

	"github.com/markoxley/daggermind/neuralnetwork/config"
	"github.com/markoxley/daggermind/neuralnetwork/matrix"
)

func TestNewNetwork(t *testing.T) {
	type args struct {
		topology     []uint32
		learningRate float64
	}
	tests := []struct {
		name    string
		args    args
		want    *Network
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := New(&config.NetworkConfiguration{
				Topology:     tt.args.topology,
				LearningRate: tt.args.learningRate,
			})
			if (err != nil) != tt.wantErr {
				t.Errorf("New() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("New() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNetwork_FeedForward(t *testing.T) {
	type fields struct {
		topology       []uint32
		weightMatrices []*matrix.Matrix
		valueMatrices  []*matrix.Matrix
		biasMatrices   []*matrix.Matrix
		learningRate   float64
	}
	type args struct {
		input []float64
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &Network{
				topology:       tt.fields.topology,
				weightMatrices: tt.fields.weightMatrices,
				valueMatrices:  tt.fields.valueMatrices,
				biasMatrices:   tt.fields.biasMatrices,
				learningRate:   tt.fields.learningRate,
			}
			if err := s.feedForward(tt.args.input); (err != nil) != tt.wantErr {
				t.Errorf("Network.FeedForward() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
