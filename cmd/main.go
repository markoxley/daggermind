package cmd

import (
	"fmt"
	"log"

	"math/rand"

	"github.com/markoxley/daggermind/neuralnetwork/network"
)

// Run is the main execution point
func Run() error {

	top := []uint32{2, 5, 5, 3}
	snn, _ := network.New(top, 0.1)
	targetInputs := [][]float64{
		{0, 0},
		{1, 1},
		{1, 0},
		{0, 1},
	}
	targetOutputs := [][]float64{
		{0, 0, 0},
		{1, 1, 0},
		{0, 1, 1},
		{0, 1, 1},
	}

	epoch := 100_000

	fmt.Println("training started")

	for i := 0; i < epoch; i++ {
		index := rand.Int() % 4
		if err := snn.FeedForward(targetInputs[index]); err != nil {
			log.Fatalf("training error: %v\n", err)
		}
		//fmt.Printf("Inputs: %v, Target: %v, Result: %v\n", targetInputs[index], targetOutputs[index], snn.GetPrediction())
		if err := snn.BackPropagate(targetOutputs[index]); err != nil {
			log.Fatalf("training error: %v\n", err)
		}

	}
	fmt.Println("training completed")

	for _, in := range targetInputs {
		snn.FeedForward(in)
		preds := snn.GetPrediction()
		fmt.Printf("%v : %v\n", in, preds)
	}
	return nil
}
