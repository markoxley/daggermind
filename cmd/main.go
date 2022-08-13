package cmd

import (
	"fmt"
	"log"
	"math"

	"github.com/markoxley/daggermind/neuralnetwork/config"
	"github.com/markoxley/daggermind/neuralnetwork/network"
	"github.com/markoxley/daggermind/neuralnetwork/train"
)

var (
	animals      = []string{"human", "dog", "cat", "bird", "lizard", "bat", "spider", "fly", "kangaroo", "fish", "whale"}
	attributes   = []string{"legs", "arms", "speaks", "barks", "purrs", "hair", "flies", "wings", "eggs", "pouch", "scales", "swims", "lungs"}
	targetInputs = [][]float64{
		// legs, arms, speaks, woofs, purrs, fur, flies, wings, eggs, pouch, scales, swims, lungs
		/* human    */ {2, 2, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1},
		/* dog      */ {4, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1},
		/* cat      */ {4, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1},
		/* bird     */ {2, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 1},
		/* lizard   */ {4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1},
		/* bat      */ {2, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 1},
		/* spider   */ {8, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
		/* fly      */ {6, 0, 0, 0, 0, 0, 1, 4, 1, 0, 0, 0, 0},
		/* kangaroo */ {2, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1},
		/* fish     */ {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0},
		/* whale    */ {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1},
	}
	targetOutputs = [][]float64{
		// human, dog, cat, bird, lizard, bat, spider, fly, kangaroo, fish,whale
		{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
		{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
	}
	animalCount = len(targetInputs)
)

// Run is the main execution point
func Run() error {
	config := config.New([]uint32{13, 16, 11})
	config.Quiet = false
	snn, _ := network.New(config)

	for i := 0; i < 2; i++ {
		targetInputs = append(targetInputs, targetInputs...)
		targetOutputs = append(targetOutputs, targetOutputs...)
	}

	td := train.New(10_000, 0.5, 0.01)

	for i := 0; i < len(targetInputs); i++ {
		td.AddRow(targetInputs[i], targetOutputs[i])
	}

	if _, err := snn.Train(td); err != nil {
		return fmt.Errorf("training error: %v", err)
	}

	for i, d := range targetInputs[:animalCount] {
		testData(snn, animals[i], d)
	}

	fmt.Println()
	testData(snn, "Osterich", []float64{2, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1})
	testData(snn, "Flea", []float64{6, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0})
	testData(snn, "Siri", []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	testData(snn, "Platypus", []float64{4, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1})
	return nil
}

func testData(snn *network.Network, nm string, in []float64) {

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
	an := fmt.Sprintf("%v (%v%%)", animals[x], math.Floor(h*100))
	if h < .85 {
		an = fmt.Sprintf("Not sure, maybe a %v", an)
	}
	if h < .5 {
		an = "I have no idea"
	}
	fmt.Printf("%v: %v = %v \n", nm, m, an)
}
