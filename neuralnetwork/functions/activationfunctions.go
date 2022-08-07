package functions

import (
	"math"
	"math/rand"
)

type NeuralFunction func(v float64) float64

// ApplyRandom returns a random value
func ApplyRandom(v float64) float64 {
	return rand.Float64()
}

// Sigmoid returns the sigmoid value of the argument
func Sigmoid(v float64) float64 {
	return 1.0 / (1 + math.Exp(-v))
}

// Dsigmoid returns derivative of sigmoid function
func Dsigmoid(v float64) float64 {
	return v * (1 - v)
}

func Tester(v float64) float64 {
	if v <= 0 {
		return 0
	}
	return v
}
