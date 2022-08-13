package matrix

import (
	"math"
	"reflect"
	"testing"

	"github.com/markoxley/daggermind/neuralnetwork/functions"
)

// Tests

func TestNew(t *testing.T) {
	type args struct {
		cols uint32
		rows uint32
	}
	tests := []struct {
		name string
		args args
		want *Matrix
	}{
		{
			name: "Test 2 x 2",
			args: args{2, 2},
			want: &Matrix{2, 2, make([]float64, 4)},
		},
		{
			name: "Test 3 x 4",
			args: args{3, 4},
			want: &Matrix{3, 4, make([]float64, 12)},
		},
		{
			name: "Test 4 x 3",
			args: args{4, 3},
			want: &Matrix{4, 3, make([]float64, 12)},
		},
		{
			name: "Test 10 x 1",
			args: args{10, 1},
			want: &Matrix{10, 1, make([]float64, 10)},
		},
		{
			name: "Test 1 x 10",
			args: args{1, 10},
			want: &Matrix{1, 10, make([]float64, 10)},
		},
		{
			name: "Test 0 x 0",
			args: args{0, 0},
			want: &Matrix{0, 0, make([]float64, 0)},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := New(tt.args.cols, tt.args.rows); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("New() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_Cols(t *testing.T) {
	type args struct {
		cols uint32
		rows uint32
	}
	tests := []struct {
		name string
		args args
		want uint32
	}{
		{
			name: "Test 2 x 2",
			args: args{2, 2},
			want: 2,
		},
		{
			name: "Test 3 x 4",
			args: args{3, 4},
			want: 3,
		},
		{
			name: "Test 4 x 3",
			args: args{4, 3},
			want: 4,
		},
		{
			name: "Test 10 x 1",
			args: args{10, 1},
			want: 10,
		},
		{
			name: "Test 1 x 10",
			args: args{1, 10},
			want: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := New(tt.args.cols, tt.args.rows)
			if got := m.Cols(); got != tt.want {
				t.Errorf("Matrix.Cols() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_Rows(t *testing.T) {
	type args struct {
		cols uint32
		rows uint32
	}
	tests := []struct {
		name string
		args args
		want uint32
	}{
		{
			name: "Test 2 x 2",
			args: args{2, 2},
			want: 2,
		},
		{
			name: "Test 3 x 4",
			args: args{3, 4},
			want: 4,
		},
		{
			name: "Test 4 x 3",
			args: args{4, 3},
			want: 3,
		},
		{
			name: "Test 10 x 1",
			args: args{10, 1},
			want: 1,
		},
		{
			name: "Test 1 x 10",
			args: args{1, 10},
			want: 10,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := New(tt.args.cols, tt.args.rows)
			if got := m.Rows(); got != tt.want {
				t.Errorf("Matrix.Rows() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_Values(t *testing.T) {
	type args struct {
		cols uint32
		rows uint32
	}
	tests := []struct {
		name string
		args args
		want []float64
	}{
		{
			name: "Test 2 x 2",
			args: args{2, 2},
			want: []float64{0, 0, 0, 0},
		},
		{
			name: "Test 3 x 4",
			args: args{3, 4},
			want: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name: "Test 4 x 3",
			args: args{4, 3},
			want: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name: "Test 10 x 1",
			args: args{10, 1},
			want: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name: "Test 1 x 10",
			args: args{1, 10},
			want: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := New(tt.args.cols, tt.args.rows)
			if got := m.Values(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Matrix.Values() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_At(t *testing.T) {
	type fields struct {
		cols   uint32
		rows   uint32
		values []float64
	}
	type args struct {
		col uint32
		row uint32
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    float64
		wantErr bool
	}{
		{
			name:    "Test 2 x 2, position 0,0",
			args:    args{0, 0},
			fields:  fields{2, 2, []float64{1, 2, 3, 4}},
			want:    1,
			wantErr: false,
		},
		{
			name:    "Test 3 x 4, position 2,1",
			args:    args{2, 1},
			fields:  fields{3, 4, []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}},
			want:    12,
			wantErr: false,
		},
		{
			name:    "Test 4 x 3, position 3,2",
			args:    args{3, 2},
			fields:  fields{4, 3, []float64{100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89}},
			want:    89,
			wantErr: false,
		},
		{
			name:    "Test 10 x 1, position 1,0",
			args:    args{1, 0},
			fields:  fields{10, 1, []float64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}},
			want:    9,
			wantErr: false,
		},
		{
			name:    "Test 1 x 10, position 0,2",
			args:    args{0, 2},
			fields:  fields{1, 10, []float64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}},
			want:    8,
			wantErr: false,
		},
		{
			name:    "Test 1 x 10, position 1,2",
			args:    args{1, 2},
			fields:  fields{1, 10, []float64{10, 9, 8, 7, 6, 5, 4, 3, 2, 1}},
			want:    0,
			wantErr: true,
		},
		{
			name:    "Test 4 x 3, position 4,2",
			args:    args{4, 2},
			fields:  fields{4, 3, []float64{100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89}},
			want:    0,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix{
				cols:   tt.fields.cols,
				rows:   tt.fields.rows,
				values: tt.fields.values,
			}
			got, err := m.At(tt.args.col, tt.args.row)
			if (err != nil) != tt.wantErr {
				t.Errorf("Matrix.At() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Matrix.At() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_Multiply(t *testing.T) {
	type fields struct {
		cols   uint32
		rows   uint32
		values []float64
	}
	type args struct {
		tgt *Matrix
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    *Matrix
		wantErr bool
	}{
		{
			name:    "Test 2 x 2 * 3 x 2",
			args:    args{New(3, 2)},
			fields:  fields{2, 2, []float64{1, 2, 3, 4}},
			want:    &Matrix{3, 2, []float64{0, 0, 0, 0, 0, 0}},
			wantErr: false,
		},
		{
			name:    "Test 2 x 2 * 2 x 3",
			args:    args{New(2, 3)},
			fields:  fields{2, 2, []float64{1, 2, 3, 4}},
			want:    nil,
			wantErr: true,
		},
		{
			name:    "Test 3 x 4 * 4 x 3",
			args:    args{&Matrix{4, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}},
			fields:  fields{3, 4, []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}},
			want:    &Matrix{4, 4, []float64{76.0, 88.0, 100.0, 112.0, 166.0, 196.0, 226.0, 256.0, 256.0, 304.0, 352.0, 400.0, 346.0, 412.0, 478.0, 544.0}},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix{
				cols:   tt.fields.cols,
				rows:   tt.fields.rows,
				values: tt.fields.values,
			}
			got, err := m.Multiply(tt.args.tgt)
			if (err != nil) != tt.wantErr {
				t.Errorf("Matrix.Multiply() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Matrix.Multiply() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_MultiplyScalar(t *testing.T) {
	type fields struct {
		cols   uint32
		rows   uint32
		values []float64
	}
	type args struct {
		v float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *Matrix
	}{
		{
			name:   "Test 2 x 2 * 0.5",
			args:   args{0.5},
			fields: fields{2, 2, []float64{1, 2, 3, 4}},
			want:   &Matrix{2, 2, []float64{0.5, 1, 1.5, 2}},
		},
		{
			name:   "Test 2 x 2 * 2",
			args:   args{2},
			fields: fields{2, 2, []float64{1, 2, 3, 4}},
			want:   &Matrix{2, 2, []float64{2, 4, 6, 8}},
		},
		{
			name:   "Test 3 x 4 * 4",
			args:   args{4},
			fields: fields{3, 4, []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}},
			want:   &Matrix{3, 4, []float64{8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix{
				cols:   tt.fields.cols,
				rows:   tt.fields.rows,
				values: tt.fields.values,
			}
			got := m.MultiplyScalar(tt.args.v)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Matrix.MultiplyScalar() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_Add(t *testing.T) {
	type fields struct {
		cols   uint32
		rows   uint32
		values []float64
	}
	type args struct {
		tgt *Matrix
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    *Matrix
		wantErr bool
	}{
		{
			name:    "Test 2 x 2 + 3 x 2",
			args:    args{New(3, 2)},
			fields:  fields{2, 2, []float64{1, 2, 3, 4}},
			want:    nil,
			wantErr: true,
		},
		{
			name:    "Test 2 x 2 + 2 x 3",
			args:    args{New(2, 3)},
			fields:  fields{2, 2, []float64{1, 2, 3, 4}},
			want:    nil,
			wantErr: true,
		},
		{
			name:    "Test 3 x 4 + 3 x 4",
			args:    args{&Matrix{3, 4, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}},
			fields:  fields{3, 4, []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}},
			want:    &Matrix{3, 4, []float64{3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36}},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix{
				cols:   tt.fields.cols,
				rows:   tt.fields.rows,
				values: tt.fields.values,
			}
			got, err := m.Add(tt.args.tgt)
			if (err != nil) != tt.wantErr {
				t.Errorf("Matrix.Add() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Matrix.Add() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_AddScalar(t *testing.T) {
	type fields struct {
		cols   uint32
		rows   uint32
		values []float64
	}
	type args struct {
		v float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *Matrix
	}{
		{
			name:   "Test 2 x 2 + 0.5",
			args:   args{0.5},
			fields: fields{2, 2, []float64{1, 2, 3, 4}},
			want:   &Matrix{2, 2, []float64{1.5, 2.5, 3.5, 4.5}},
		},
		{
			name:   "Test 2 x 2 + 2",
			args:   args{2},
			fields: fields{2, 2, []float64{1, 2, 3, 4}},
			want:   &Matrix{2, 2, []float64{3, 4, 5, 6}},
		},
		{
			name:   "Test 3 x 4 + 4",
			args:   args{4},
			fields: fields{3, 4, []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}},
			want:   &Matrix{3, 4, []float64{6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix{
				cols:   tt.fields.cols,
				rows:   tt.fields.rows,
				values: tt.fields.values,
			}
			if got := m.AddScalar(tt.args.v); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Matrix.AddScalar() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_Negative(t *testing.T) {
	type fields struct {
		cols   uint32
		rows   uint32
		values []float64
	}
	tests := []struct {
		name   string
		fields fields
		want   *Matrix
	}{
		{
			name:   "Test 2 x 2",
			fields: fields{2, 2, []float64{-1, -2, -3, -4}},
			want:   &Matrix{2, 2, []float64{1, 2, 3, 4}},
		},
		{
			name:   "Test 2 x 3",
			fields: fields{2, 3, []float64{1, 2, 3, 4, 5, 6}},
			want:   &Matrix{2, 3, []float64{-1, -2, -3, -4, -5, -6}},
		},
		{
			name:   "Test 4 x 3",
			fields: fields{3, 4, []float64{2, -4, 6, -8, 10, -12, 14, -16, 18, -20, 22, -24}},
			want:   &Matrix{3, 4, []float64{-2, 4, -6, 8, -10, 12, -14, 16, -18, 20, -22, 24}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix{
				cols:   tt.fields.cols,
				rows:   tt.fields.rows,
				values: tt.fields.values,
			}
			if got := m.Negative(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Matrix.Negative() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_Transpose(t *testing.T) {
	type fields struct {
		cols   uint32
		rows   uint32
		values []float64
	}
	tests := []struct {
		name   string
		fields fields
		want   *Matrix
	}{
		{
			name:   "Test 2 x 2",
			fields: fields{2, 2, []float64{-1, -2, -3, -4}},
			want:   &Matrix{2, 2, []float64{-1, -3, -2, -4}},
		},
		{
			name:   "Test 2 x 3",
			fields: fields{2, 3, []float64{1, 2, 3, 4, 5, 6}},
			want:   &Matrix{3, 2, []float64{1.0, 3.0, 5.0, 2.0, 4.0, 6.0}},
		},
		{
			name:   "Test 4 x 3",
			fields: fields{3, 4, []float64{2, -4, 6, -8, 10, -12, 14, -16, 18, -20, 22, -24}},
			want:   &Matrix{4, 3, []float64{2.0, -8.0, 14.0, -20.0, -4.0, 10.0, -16.0, 22.0, 6.0, -12.0, 18.0, -24.0}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix{
				cols:   tt.fields.cols,
				rows:   tt.fields.rows,
				values: tt.fields.values,
			}
			if got := m.Transpose(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Matrix.Transpose() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_ApplyFunction(t *testing.T) {
	type fields struct {
		cols   uint32
		rows   uint32
		values []float64
	}
	type args struct {
		f functions.NeuralFunction
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   *Matrix
	}{
		{
			name:   "Add 1",
			fields: fields{2, 3, []float64{2, 4, 6, 8, 10, 12}},
			args: args{func(v float64) float64 {
				return v + 1
			}},
			want: &Matrix{2, 3, []float64{3, 5, 7, 9, 11, 13}},
		},
		{
			name:   "Multiply 10",
			fields: fields{3, 3, []float64{3, 6, 3, 2, 8, 9, 7, 4, 3}},
			args: args{func(v float64) float64 {
				return v * 10
			}},
			want: &Matrix{3, 3, []float64{30, 60, 30, 20, 80, 90, 70, 40, 30}},
		},
		{
			name:   "Square",
			fields: fields{3, 1, []float64{3, 6, 9}},
			args: args{func(v float64) float64 {
				return math.Pow(v, 2)
			}},
			want: &Matrix{3, 1, []float64{9, 36, 81}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix{
				cols:   tt.fields.cols,
				rows:   tt.fields.rows,
				values: tt.fields.values,
			}
			if got := m.ApplyFunction(tt.args.f); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Matrix.ApplyFunction() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_Set(t *testing.T) {
	type fields struct {
		cols   uint32
		rows   uint32
		values []float64
	}
	type args struct {
		col uint32
		row uint32
		v   float64
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
		want    *Matrix
	}{
		{
			name:    "2 x 2 Set 1 x 1",
			fields:  fields{2, 2, []float64{0, 0, 0, 0}},
			args:    args{1, 1, 10},
			wantErr: false,
			want:    &Matrix{2, 2, []float64{0, 0, 0, 10}},
		},
		{
			name:    "2 x 2 Set 0 x 0",
			fields:  fields{2, 2, []float64{0, 0, 0, 0}},
			args:    args{0, 0, 5},
			wantErr: false,
			want:    &Matrix{2, 2, []float64{5, 0, 0, 0}},
		},
		{
			name:    "3 x 3 Set 3 x 2",
			fields:  fields{3, 3, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0}},
			args:    args{3, 2, 10},
			wantErr: true,
			want:    &Matrix{3, 3, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix{
				cols:   tt.fields.cols,
				rows:   tt.fields.rows,
				values: tt.fields.values,
			}
			if err := m.Set(tt.args.col, tt.args.row, tt.args.v); (err != nil) != tt.wantErr {
				t.Errorf("Matrix.Set() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !reflect.DeepEqual(m, tt.want) {
				t.Errorf("Matrix.Set() = %v, want %v", m, tt.want)
			}
		})
	}
}

func TestNewFromSlice(t *testing.T) {
	type args struct {
		slc []float64
	}
	tests := []struct {
		name string
		args args
		want *Matrix
	}{
		{
			name: "3 element sequential",
			args: args{[]float64{1, 2, 3}},
			want: &Matrix{
				cols:   3,
				rows:   1,
				values: []float64{1, 2, 3},
			},
		},
		{
			name: "5 element random",
			args: args{[]float64{3.27, 9.543, 12.549, 10.44541, 2.4321}},
			want: &Matrix{
				cols:   5,
				rows:   1,
				values: []float64{3.27, 9.543, 12.549, 10.44541, 2.4321},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewFromSlice(tt.args.slc); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewFromSlice() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMatrix_SetValues(t *testing.T) {
	type fields struct {
		cols   uint32
		rows   uint32
		values []float64
	}
	type args struct {
		vals []float64
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
		want    *Matrix
	}{
		{
			name:    "3x3 - correct size",
			fields:  fields{3, 3, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0}},
			args:    args{[]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}},
			wantErr: false,
			want: &Matrix{
				cols:   3,
				rows:   3,
				values: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
			},
		},
		{
			name:    "3x3 - too large",
			fields:  fields{3, 3, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0}},
			args:    args{[]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
			wantErr: true,
			want: &Matrix{
				cols:   3,
				rows:   3,
				values: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0},
			},
		},
		{
			name:    "3x3 - too small",
			fields:  fields{3, 3, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0}},
			args:    args{[]float64{1, 2, 3, 4, 5, 6, 7, 8}},
			wantErr: true,
			want: &Matrix{
				cols:   3,
				rows:   3,
				values: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0},
			},
		},
		{
			name:    "3x3 - nil",
			fields:  fields{3, 3, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0}},
			args:    args{nil},
			wantErr: true,
			want: &Matrix{
				cols:   3,
				rows:   3,
				values: []float64{0, 0, 0, 0, 0, 0, 0, 0, 0},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix{
				cols:   tt.fields.cols,
				rows:   tt.fields.rows,
				values: tt.fields.values,
			}
			if err := m.SetValues(tt.args.vals); (err != nil) != tt.wantErr {
				t.Errorf("Matrix.SetValues() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !reflect.DeepEqual(m, tt.want) {
				t.Errorf("Matrix.SetValues() = %v, want %v", m, tt.want)
			}
		})
	}
}
