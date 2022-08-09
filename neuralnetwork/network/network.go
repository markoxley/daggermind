package network

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/markoxley/daggermind/neuralnetwork/config"
	"github.com/markoxley/daggermind/neuralnetwork/functions"
	"github.com/markoxley/daggermind/neuralnetwork/matrix"
	"github.com/markoxley/daggermind/neuralnetwork/train"
)

type SaveData struct {
	Topology       []uint32           `json:"t"`
	WeightMatrices []*matrix.SaveData `json:"w"`
	BiasMatrices   []*matrix.SaveData `json:"b"`
	LearningRate   float64            `json:"l"`
}
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
func New(c *config.NetworkConfiguration) (*Network, error) {
	s := Network{
		topology:     c.Topology,
		learningRate: c.LearningRate,
	}
	for i := 0; i < len(s.topology)-1; i++ {
		wm := matrix.New(s.topology[i+1], s.topology[i])
		s.weightMatrices = append(s.weightMatrices, wm.ApplyFunction(functions.ApplyRandom))

		bm := matrix.New(s.topology[i+1], 1)
		s.biasMatrices = append(s.biasMatrices, bm.ApplyFunction(functions.ApplyRandom))
	}
	s.valueMatrices = make([]*matrix.Matrix, len(s.topology))
	return &s, nil
}

// FeedForward runs the network forwards
func (n *Network) feedForward(input []float64) error {
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

func (n *Network) backPropagate(tgtOut []float64) error {
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

func (n *Network) getPrediction() []float64 {
	return n.valueMatrices[len(n.valueMatrices)-1].Values()
}

func (n *Network) Train(td *train.Data) (float64, error) {
	td.Prepare()
	for i := 0; i < int(td.Iterations); i++ {
		row := td.RandomTrainingRow()
		if err := n.feedForward(row.Input); err != nil {
			return 0, fmt.Errorf("training error: %v", err)
		}
		if err := n.backPropagate(row.Ouput); err != nil {
			return 0, fmt.Errorf("training error: %v", err)
		}
	}
	testData := td.TestData()
	var errVal float64
	for _, test := range testData {
		answer, err := n.Predict(test.Input)
		if err != nil {
			return 0, fmt.Errorf("error testing training data: %v", err)
		}
		var v float64
		for i, a := range answer {
			v += math.Pow(test.Ouput[i]-a, 2)
		}
		errVal += v / float64(len(answer))
	}
	return math.Sqrt(errVal / float64(len(testData))), nil
}

func (n *Network) Predict(input []float64) ([]float64, error) {
	err := n.feedForward(input)
	if err != nil {
		return nil, fmt.Errorf("prediction error: %v", err)
	}
	return n.getPrediction(), nil
}

func (n *Network) ToSaveData() *SaveData {
	sd := SaveData{
		Topology:       n.topology,
		LearningRate:   n.learningRate,
		WeightMatrices: make([]*matrix.SaveData, len(n.weightMatrices)),
		BiasMatrices:   make([]*matrix.SaveData, len(n.biasMatrices)),
	}
	for i, wm := range n.weightMatrices {
		sd.WeightMatrices[i] = wm.ToSaveData()
	}
	for i, bm := range n.biasMatrices {
		sd.BiasMatrices[i] = bm.ToSaveData()
	}
	return &sd
}

func (n *Network) ToJson() ([]byte, error) {
	return json.Marshal(n.ToSaveData())
}

func (n *Network) Write(w io.Writer) error {
	j, err := n.ToJson()
	if err != nil {
		return fmt.Errorf("network write error: %v", err)
	}
	c, err := w.Write(j)
	if err != nil {
		return fmt.Errorf("network write error: %v", err)
	}
	if c != len(j) {
		return errors.New("incorrect number of bytes written")
	}
	return nil
}

func (n *Network) SaveToFile(fp string) error {
	j, err := n.ToJson()
	if err != nil {
		return fmt.Errorf("error saving data: %v", err)
	}
	return os.WriteFile(fp, j, os.ModePerm)
}

func FromJson(b []byte) (*Network, error) {
	sd := SaveData{}
	err := json.Unmarshal(b, &sd)
	if err != nil {
		return nil, fmt.Errorf("network unmarshal error: %v", err)
	}

	return FromSaveData(&sd)
}

func FromSaveData(sd *SaveData) (*Network, error) {
	if sd == nil {
		return nil, errors.New("missing save data")
	}
	weightMatrices := make([]*matrix.Matrix, len(sd.WeightMatrices))
	for i, wsd := range sd.WeightMatrices {
		wm, err := matrix.FromSaveData(wsd)
		if err != nil {
			return nil, fmt.Errorf("unable to apply weight matrix: %v", err)
		}
		weightMatrices[i] = wm
	}

	biasMatrices := make([]*matrix.Matrix, len(sd.BiasMatrices))
	for i, bsd := range sd.BiasMatrices {
		bm, err := matrix.FromSaveData(bsd)
		if err != nil {
			return nil, fmt.Errorf("unable to apply bias matrix: %v", err)
		}
		biasMatrices[i] = bm
	}

	valueMatrices := make([]*matrix.Matrix, len(sd.Topology))
	for i, t := range sd.Topology {
		valueMatrices[i] = matrix.New(t, 1)
	}
	n := Network{
		topology:       sd.Topology,
		learningRate:   sd.LearningRate,
		weightMatrices: weightMatrices,
		valueMatrices:  valueMatrices,
		biasMatrices:   biasMatrices,
	}

	return &n, nil
}

func Read(r io.Reader) (*Network, error) {
	b := make([]byte, 0, 64)
	res := make([]byte, 0)
	t := 0
	for {
		c, err := r.Read(b)
		if err != nil {
			return nil, fmt.Errorf("read error: %v", err)
		}
		if c > 0 {
			t += c
			res = append(res, b[:c]...)
		}
		if c < len(b) {
			break
		}

	}
	return FromJson(res)
}

func FromFile(fp string) (*Network, error) {
	b, err := os.ReadFile(fp)
	if err != nil {
		return nil, fmt.Errorf("unable to read data: %v", err)
	}
	return FromJson(b)
}
