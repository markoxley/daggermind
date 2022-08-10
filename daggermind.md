# DaggerMind

## Basic artificial neural network

### basic example

```
    // Create configuration for the network
    // In this case, we have 13 input neurons, 11
    // output neurons and one hidden layer with 16 neurons
    config := config.New([]uint32{13, 16, 11})

    // Create a new instance of the neural network,
    // passing the configurations
    nn, _ := network.New(config)

    // Initialis a training data set
    // This data set will attempt 100,000 iterations,
    // using 50% of the test data, and a target error
    // tolerance of 0.1
    td := train.New(100_000, 0.5, 0.1)


    // Add the training and test data to the data set
    for i := 0; i < len(targetInputs); i++ {
    	td.AddRow(targetInputs[i], targetOutputs[i])
    }

    // Train the network, handling any errors
    if _, err := snn.Train(td); err != nil {
		return fmt.Errorf("training error: %v", err)
	}

    // Get a prediction from the network, again, be
    // mindful of any returned errors
    pr, err := snn.Predict(in)
```
