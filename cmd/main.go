package cmd

import (
	"fmt"
	"log"
	"math"

	"github.com/markoxley/daggermind/neuralnetwork/config"
	"github.com/markoxley/daggermind/neuralnetwork/functions"
	"github.com/markoxley/daggermind/neuralnetwork/network"
	"github.com/markoxley/daggermind/neuralnetwork/train"
)

// Run is the main execution point
func Run() error {
	config := &config.NetworkConfiguration{
		Topology:     []uint32{9, 14, 14, 9},
		LearningRate: 0.1,
		Functions:    functions.Sigmoid,
	}

	snn, _ := network.New(config)
	targetInputs := [][]float64{
		// legs, arms, speaks, woofs, purrs, fur, flies, wings, eggs
		{2, 2, 1, 0, 0, 1, 0, 0, 0},
		{4, 0, 0, 1, 0, 1, 0, 0, 0},
		{4, 0, 0, 0, 1, 1, 0, 0, 0},
		{2, 0, 0, 0, 0, 0, 1, 2, 1},
		{4, 0, 0, 0, 0, 0, 0, 0, 1},
		{2, 0, 0, 0, 0, 1, 1, 2, 0},
		{8, 0, 0, 0, 0, 0, 0, 0, 1},
		{6, 0, 0, 0, 0, 0, 1, 4, 1},
		{2, 2, 0, 0, 0, 1, 0, 0, 0},
	}
	targetOutputs := [][]float64{
		// human, dog, cat, bird, lizard, bat, spider, fly, kangaroo
		{1, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 1, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 1, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 1, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 1, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 1, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 1},
	}

	for i := 0; i < 3; i++ {
		targetInputs = append(targetInputs, targetInputs...)
		targetOutputs = append(targetOutputs, targetOutputs...)
	}
	td := &train.Data{
		Split:      0.7,
		Iterations: 100000,
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
	animals := []string{"human", "dog", "cat", "bird", "lizard", "bat", "spider", "fly", "kangaroo"}
	for i, d := range targetInputs[:8] {
		testData(snn, animals[i], d)
	}
	fmt.Println()
	testData(snn, "Osterich", []float64{2, 0, 0, 0, 0, 0, 0, 2, 1})
	testData(snn, "Flea", []float64{6, 0, 0, 0, 0, 0, 0, 0, 1})
	testData(snn, "Siri", []float64{0, 0, 1, 0, 0, 0, 0, 0, 0})
	testData(snn, "Platypus", []float64{4, 0, 0, 0, 0, 1, 0, 0, 1})
	return nil
}

func testData(snn *network.Network, nm string, in []float64) {
	animals := []string{"human", "dog", "cat", "bird", "lizard", "bat", "spider", "fly"}
	attributes := []string{"legs", "arms", "speaks", "barks", "purrs", "hair", "flies", "wings"}

	pr, err := snn.Predict(in)
	if err != nil {
		log.Fatal(err)
	}
	x := -1
	h := float64(0)
	for j, s := range pr {
		if s > h {
			h = s
			x = j
		}
	}
	m := ""
	for i, a := range attributes {
		if in[i] > 0 {
			if m != "" {
				m += ", "
			}
			m += fmt.Sprintf("%v (%v)", a, in[i])
		}
	}
	an := animals[x]
	if h < .8 {
		an = fmt.Sprintf("Not sure, maybe a %v", an)
	}
	fmt.Printf("%v: %v = %v (%v%%)\n", nm, m, an, math.Floor(h*100))
}
