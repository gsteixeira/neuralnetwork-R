# A Neural Network in R

A simple neural network implementation in R


## usage:

```shell
    Rscript neural.R
    # or
    make
```

## Create a neural network:

Create a network telling the size (nodes) of earch layer.
```R
    input_size <- 2
    output_size <- 1
    # training parameters
    inputs <- array(c(0, 1, 0, 1, 0, 0, 1, 1), dim=c(4, input_size))
    outputs <- array(c(0.0, 1.0, 1.0, 0.0), dim=c(4, output_size))
    # Create the network
    nn <- NewNeuralNetwork(num_input_layer, output_size, num_hidden_nodes)
    # Train
    iteracions <- 10000
    nn <- train(nn, inputs, outputs, iteracions)
    # now make some predictions
    predicted <- predict(nn, c(1, 0));
    print(predicted)
    # ~ [0.9]
```



## to run tests:

```shell
    Rscript neural.R test
    # or
    make test
```
