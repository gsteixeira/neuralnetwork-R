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
```c
    NeuralNetwork nn;
    nn = NewNeuralNetwork(input_size, output_size, hidden_layer_size);
    // Train it
    train(nn, input_size, output_size, inputs, outputs, 10000);
    // Now, make predictions
    double foo[2] = {0, 1}; // should result in 1
    predicted = predict(nn, foo);
    printf("predicted: %f \n", predicted);
    // predicted: ~ 0.99;
```



## to run tests:

```shell
    Rscript neural.R test
    # or
    make test
```
