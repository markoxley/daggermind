package matrix

import "testing"

// Benchmarks

func BenchmarkNew(b *testing.B) {
	for i := 0; i < b.N; i++ {
		New(20, 200)
	}
}
