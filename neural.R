# 
# 
# Implementation of a simple Feedforward Neural Network in R
# 
#     Author: Gustavo Selbach Teixeira (gsteixei@gmail.com)
# 
#

# The Layer object class
setClass("Layer",
         slots=list(values="vector",
                    bias="vector",
                    deltas="vector",
                    weights="matrix"))

# NeuralNetwork object class
setClass("NeuralNetwork",
         slots=list(input_layer="Layer",
                    hidden_layer="Layer",
                    output_layer="Layer"))
        
# Layer constructor method
NewLayer<-function(n_nodes, n_synapses) {
    weight_size <- n_nodes * n_synapses
    l <- new("Layer",
             values=c(as.double(runif(n_nodes, 0, 1))),
             bias=c(as.double(runif(n_nodes, 0, 1))),
             deltas=c(as.double(runif(n_nodes, 0, 1))),
             weights=array(as.double(runif(weight_size, 0, 1)),
                           dim=c(n_synapses, n_nodes))
             )
    return(l)
}
                    
# NeuralNetwork constructor method
NewNeuralNetwork<-function(inputs, outputs, hidden_size) {
    nn <- new("NeuralNetwork",
              input_layer=NewLayer(inputs, 0),
              hidden_layer=NewLayer(hidden_size, inputs),
              output_layer=NewLayer(outputs, hidden_size))
    return(nn)
}

# The sigmoid logistical function
sigmoid<-function(x){
    return (1 / (1 + exp(-x)))
}

# The derivative sigmoid logistical function
d_sigmoid<-function(x){
    return (x * (1 - x))
}

# The activation function
activation_function<-function(source, target) {
    for (j in 1:length(target@values)) {
        activation = target@bias[j]
        for (k in 1:length(source@values)) {
            activation <- (activation 
                            + (source@values[k] * target@weights[k, j]))
        }
        target@values[j] = sigmoid(activation)
    }
    return(target@values)
}

# Compute the deltas though the network
calc_deltas<-function(source, target) {
    num_nodes_target <- length(target@values)
    num_nodes_source <- length(source@values)
    delta = c()
    for (j in 1:num_nodes_target){
        error = 0.0
        for (i in 1:num_nodes_source) {
            error <- error + (source@deltas[i] * source@weights[j, i])
        }
        delta[j] <- (error * d_sigmoid(target@values[j]))
    }
    return(delta)
}

# Compute the deltas for the output layer
calc_delta_output<-function(expected, nn) {
    delta_output <- c()
    for (j in 1:length(nn@output_layer@values)) {
        errors <- (expected[j] - nn@output_layer@values[j])
        delta_output[j] <- (errors 
                            * d_sigmoid(nn@output_layer@values[j]))
    }
    return(delta_output)
}

# Update the weights of synapses 
update_weights<-function(source, target, learning_rate) {
    dest_number = length(target@values)
    for (j in 1:length(source@values)) {
        source@bias[j] <- (source@bias[j] 
                           + (source@deltas[j] * learning_rate))
        for (k in 1:dest_number) {
            source@weights[k, j] <- (source@weights[k, j] + (target@values[k] 
                                     * source@deltas[j] * learning_rate))
        }
    }
    return(list(bias=source@bias, weights=source@weights))
}

# Neural network training loop
train<-function(nn, inputs, outputs, n_epochs) {
    learning_rate <- 0.1
    num_training_sets <- length(outputs)
    for (n in 1:n_epochs) {
        for (i in 1:num_training_sets){
            nn@input_layer@values = inputs[i,]
            # Forward pass
            nn@hidden_layer@values <- activation_function(
                                        nn@input_layer, nn@hidden_layer)
            nn@output_layer@values <- activation_function(
                                        nn@hidden_layer, nn@output_layer)
            cat(n,
                " Input: ", inputs[i,],
                " Expected: ", outputs[i,],
                " Output: ", nn@output_layer@values, "\n")
            # Back propagation
            # calculate delta for output_layer
            nn@output_layer@deltas <- calc_delta_output(outputs[i,], nn)
            nn@hidden_layer@deltas <- calc_deltas(nn@output_layer, nn@hidden_layer)
            # Update weights
            res <- update_weights(nn@output_layer, nn@hidden_layer, learning_rate)
            nn@output_layer@bias <- res$bias
            nn@output_layer@weights <- res$weights
            
            res <- update_weights(nn@hidden_layer, nn@input_layer, learning_rate)
            nn@hidden_layer@bias <- res$bias
            nn@hidden_layer@weights <- res$weights
        }
    }
    return(nn)
}

# Make a prediction. To be used after the network has been trained.
predict<-function(nn, inputs) {
    nn@input_layer@values = inputs
    # Forward pass
    nn@hidden_layer@values <- activation_function(
                                nn@input_layer, nn@hidden_layer)
    nn@output_layer@values <- activation_function(
                                nn@hidden_layer, nn@output_layer)
    return(nn@output_layer@values)
}

# The main function
main<-function() {
    inputs <- array(c(0, 1, 0, 1, 0, 0, 1, 1), dim=c(4, 2))
    outputs <- array(c(0.0, 1.0, 1.0, 0.0), dim=c(4, 1))
    
    num_hidden_nodes <- 4
    nn <- NewNeuralNetwork(2, 1, num_hidden_nodes)
    
    iteracions <- 10000
    nn <- train(nn, inputs, outputs, iteracions)
    # now make some predictions
    for (i in 1:4) {
        input <- inputs[i,]
        predicted <- predict(nn, input);
        cat("input: [", input,
            "] predicted: ", predicted,
            " expected: ", outputs[i,], "\n")
    }
}

# get command line arguments
if (length(commandArgs(trailingOnly=TRUE)) == 0) {
    main()
} else {
    args = commandArgs(trailingOnly=TRUE)
    if (args[1] == "test") {
        source("neural_test.R")
        do_tests()
    }
}
