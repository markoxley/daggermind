package config

import "github.com/markoxley/daggermind/neuralnetwork/functions"

type NetworkConfiguration struct {
	Topology     []uint32
	LearningRate float64
	Functions    functions.FunctionName
}
