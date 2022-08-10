package config

import "github.com/markoxley/daggermind/neuralnetwork/functions"

type NetworkConfiguration struct {
	Topology     []uint32
	LearningRate float64
	Functions    functions.FunctionName
	Quiet        bool
}

func New(topology []uint32) *NetworkConfiguration {
	return &NetworkConfiguration{
		Topology:     topology,
		LearningRate: 0.1,
		Functions:    functions.Sigmoid,
		Quiet:        true,
	}
}
