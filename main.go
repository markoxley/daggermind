package main

import (
	"github.com/markoxley/daggermind/cmd"
)

func main() {
            	if err := cmd.Run(); err != nil {
		log.Fatalf("unhandled error: %v", err)
	}
	log.Print("Done")

	log.Fatal()
}
