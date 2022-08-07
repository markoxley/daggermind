package network

import (
	"errors"
	"fmt"
	"math/rand"
	"time"

	"github.com/markoxley/daggermind/neuralnetwork/functions"
	"github.com/markoxley/daggermind/neuralnetwork/matrix"
)

type Network struct {
	topology       []uint32
	weightMatrices []*matrix.Matrix
	valueMatrices  []*matrix.Matrix
	biasMatrices   []*matrix.Matrix
	learningRate   float64
}

func init() {
	rand.Seed(time.Now().Unix())
}

// NewSimple creates a new simple neural network
func New(topology []uint32, learningRate float64) (*Network, error) {
	s := Network{
		topology:     topology,
		learningRate: learningRate,
	}
	for i := 0; i < len(topology)-1; i++ {
		wm := matrix.New(topology[i+1], topology[i])
		s.weightMatrices = append(s.weightMatrices, wm.ApplyFunction(functions.ApplyRandom))

		bm := matrix.New(topology[i+1], 1)
		s.biasMatrices = append(s.biasMatrices, bm.ApplyFunction(functions.ApplyRandom))
	}
	s.valueMatrices = make([]*matrix.Matrix, len(topology))
	return &s, nil
}

// FeedForward runs the network forwards
func (n *Network) FeedForward(input []float64) error {
	if len(input) != int(n.topology[0]) {
		return errors.New("incorrect input size")
	}
	values := matrix.New(uint32(len(input)), 1)
	for i, in := range input {
		values.Set(uint32(i), 0, in)
	}
	var err error
	// feed forward to next layer
	for i, w := range n.weightMatrices {
		n.valueMatrices[i] = values
		values, err = values.Multiply(w)
		if err != nil {
			return fmt.Errorf("feed forward error: %v", err)
		}
		values, err = values.Add(n.biasMatrices[i])
		if err != nil {
			return fmt.Errorf("feed forward error: %v", err)
		}
		values = values.ApplyFunction(functions.Sigmoid)
	}
	n.valueMatrices[len(n.weightMatrices)] = values

	return nil
}

func (n *Network) BackPropagate(tgtOut []float64) error {
	if len(tgtOut) != int(n.topology[len(n.topology)-1]) {
		return errors.New("output is incorrect size")
	}
	errMtx := matrix.New(uint32(len(tgtOut)), 1)
	errMtx.SetValues(tgtOut)
	errMtx, err := errMtx.Add(n.valueMatrices[len(n.valueMatrices)-1].Negative())
	if err != nil {
		return fmt.Errorf("back propagation error: %v", err)
	}
	for i := len(n.weightMatrices) - 1; i >= 0; i-- {
		prevErrors, err := errMtx.Multiply(n.weightMatrices[i].Transpose())
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}

		dOutputs := n.valueMatrices[i+1].ApplyFunction(functions.Dsigmoid)
		gradients, err := errMtx.MultiplyElements(dOutputs)
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}
		gradients = gradients.MultiplyScalar(n.learningRate)
		weightGradients, err := n.valueMatrices[i].Transpose().Multiply(gradients)
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}
		n.weightMatrices[i], err = n.weightMatrices[i].Add(weightGradients)
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}
		n.biasMatrices[i], err = n.biasMatrices[i].Add(gradients)
		if err != nil {
			return fmt.Errorf("back propagation error: %v", err)
		}
		errMtx = prevErrors
	}
	return nil

}

func (n *Network) GetPrediction() []float64 {
	return n.valueMatrices[len(n.valueMatrices)-1].Values()
}
