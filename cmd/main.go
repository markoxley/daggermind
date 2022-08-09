package cmd

import (
	"errors"
	"fmt"

	"github.com/markoxley/daggermind/neuralnetwork/config"
	"github.com/markoxley/daggermind/neuralnetwork/network"
	"github.com/markoxley/daggermind/neuralnetwork/train"
)

// Run is the main execution point
func Run() error {
	config := &config.NetworkConfiguration{
		Topology:     []uint32{2, 25, 25, 25, 25, 1},
		LearningRate: 0.1,
	}

	snn, _ := network.New(config)
	targetInputs := [][]float64{
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
		{0, 0},
		{1, 1},
		{0, 1},
		{1, 0},
	}
	targetOutputs := [][]float64{
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
		{0},
		{0},
		{1},
		{1},
	}

	td := &train.Data{
		Split:      0.7,
		Iterations: 100_000,
	}

	for i := 0; i < len(targetInputs); i++ {
		td.AddRow(targetInputs[i], targetOutputs[i])
	}

	fmt.Println("training started")
	errMargin, err := snn.Train(td)
	if err != nil {
		return fmt.Errorf("training error: %v", err)
	}
	fmt.Printf("Error margin: %v\n", errMargin)

	fmt.Println("training completed")
	fmt.Println("testing Json")
	j, err := snn.ToJson()
	if err != nil {
		return fmt.Errorf("json error: %v", err)
	}

	nn2, err := network.FromJson(j)
	if err != nil {
		return fmt.Errorf("unable to read from json: %v", err)
	}
	if nn2 == nil {
		return errors.New("json resulted in nil")
	}
	fmt.Println("network reinstated")
	j2, err := nn2.ToJson()
	if err != nil {
		return fmt.Errorf("json error (second time): %v", err)
	}

	s1 := string(j)
	s2 := string(j2)

	if s1 != s2 {
		return errors.New("networks do not match")
	}

	r1, err := snn.Predict([]float64{1, 0})
	if err != nil {
		return fmt.Errorf("unable to get first prediction: %v", err)
	}

	r2, err := nn2.Predict([]float64{1, 0})
	if err != nil {
		return fmt.Errorf("unable to get second prediction: %v", err)
	}

	for i := range r1 {
		if r1[i] != r2[i] {
			return errors.New("prediction error")
		}
	}

	fmt.Println("Testing R/W to file")
	err = snn.SaveToFile("nn.dat")
	if err != nil {
		return fmt.Errorf("saving error: %v", err)
	}

	nn3, err := network.FromFile("nn.dat")
	if err != nil {
		return fmt.Errorf("loading error: %v", err)
	}

	j3, err := nn3.ToJson()
	if err != nil {
		return fmt.Errorf("json error (third time): %v", err)
	}

	s3 := string(j3)
	if s3 != s1 {
		return errors.New("loaded network does not match")
	}

	r3, err := nn3.Predict([]float64{1, 0})
	if err != nil {
		return fmt.Errorf("unable to get third prediction: %v", err)
	}

	for i := range r1 {
		if r1[i] != r3[i] {
			return errors.New("prediction error")
		}
	}
	fmt.Println("All tests successful")
	// for i, in := range targetInputs {
	// 	v, err := snn.Predict(in)
	// 	if err != nil {
	// 		return fmt.Errorf("prediction error: %v", err)
	// 	}
	// 	fmt.Printf("%v -> %v: %v\n", in, targetOutputs[i], v)
	// }
	return nil
}
