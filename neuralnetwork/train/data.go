package train

import (
	"math"
	"math/rand"
)

type DataRow struct {
	Input []float64
	Ouput []float64
}

type Data struct {
	trainingData []*DataRow
	testingData  []*DataRow
	Data         []*DataRow
	Split        float64
	Iterations   uint32
}

func NewData(iterations uint32, split float64) *Data {
	return &Data{
		Split:      split,
		Iterations: iterations,
	}
}

func (d *Data) AddRow(inputs, output []float64) {
	d.Data = append(d.Data, &DataRow{
		Input: inputs,
		Ouput: output,
	})
}

func (d *Data) Prepare() {
	trainCount := int(math.Round(float64(len(d.Data)) * d.Split))
	testCount := len(d.Data) - trainCount
	d.trainingData = make([]*DataRow, 0, trainCount)
	d.testingData = make([]*DataRow, 0, testCount)
	index := make([]int, len(d.Data))
	for i := range index {
		index[i] = i
	}
	for i := 0; i < len(d.Data); i++ {
		p1 := rand.Intn(len(d.Data))
		p2 := rand.Intn(len(d.Data))
		tmp := index[p1]
		index[p1] = index[p2]
		index[p2] = tmp
	}
	for i, idx := range index {
		if i < trainCount {
			d.trainingData = append(d.trainingData, d.Data[idx])
		} else {
			d.testingData = append(d.testingData, d.Data[idx])
		}
	}
}

func (d *Data) RandomTrainingRow() *DataRow {
	return d.trainingData[rand.Intn(len(d.trainingData))]
}

func (d *Data) TestData() []*DataRow {
	return d.testingData
}

func (d *Data) TrainingCount() int {
	return len(d.trainingData)
}
