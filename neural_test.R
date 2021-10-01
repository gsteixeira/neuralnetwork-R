# Testing functions for neural.R
#
# Does basic tests


# Prepare a network for testing
get_nn_for_test<-function() {
    num_hidden_nodes = 2
    learning_rate = 0.1
    nn <- NewNeuralNetwork(2, 1, 2)
}

# Test the activation function
test_activation_function<-function() {
    nn <- get_nn_for_test()
    source_layer =  c(0.0, 1.0)
    target_layer =  c(0.1, 0.2)
    target_layer_bias =  c(0.1, 0.2)
    weights =  array(c(0.1, 0.3, 0.2, 0.4), dim=c(2,2))
    
    nn@input_layer@values = source_layer
    nn@hidden_layer@bias = target_layer_bias
    nn@hidden_layer@values = target_layer
    nn@hidden_layer@weights = weights
    
    nn@hidden_layer@values<-activation_function(nn@input_layer,
                                                nn@hidden_layer)
    TARGET_LAYER_FINAL = c(0.598687660112452, 0.6456563062257954) # 0,1
    for (i in 1:length(nn@hidden_layer@values)) {
        if (nn@hidden_layer@values[i] == TARGET_LAYER_FINAL[i]) {
            cat(".")
        } else {
            print("Error: test_activation_function")
            cat(nn@hidden_layer@values[i], TARGET_LAYER_FINAL[i], "\n")
        }
    }
}

# Test the update_weights function
test_update_weights<-function() {
    nn <- get_nn_for_test()
    source_bias =  c(0.1, 0.2)
    target_layer =  c(1.0, 0.0)
    weights =  array(c(0.1, 0.3, 0.2, 0.4), dim=c(2,2))
    deltas =  c(0.1, 0.2)
    ####
    nn@hidden_layer@deltas = deltas
    nn@hidden_layer@bias = source_bias
    nn@hidden_layer@weights = weights
    nn@input_layer@values = target_layer
    
    learning_rate = 0.1
    res <- update_weights(nn@hidden_layer, nn@input_layer, learning_rate)

    SOURCE_BIAS_FINAL = c(0.11000000000000001, 0.22000000000000003)
    WEIGHTS_FINAL = array(c(0.11000000000000001, 0.3,
                            0.22000000000000002, 0.4),
                          dim=c(2,2))
    for (i in 1:length(SOURCE_BIAS_FINAL)) {
        if (res$bias[i] == SOURCE_BIAS_FINAL[i]) {
            cat(".")
        } else {
            print("Error: test_update_weights - bias")
            cat(res$bias[i], SOURCE_BIAS_FINAL[i], "\n")
        }
    }
    for (i in 1:length(WEIGHTS_FINAL)) {
        for (j in 1:length(WEIGHTS_FINAL[i])) {
            if (res$weights[i][j] == WEIGHTS_FINAL[i][j]) {
                cat(".")
            } else {
                print("Error: test_update_weights - weights")
                cat(i, j, res$weights[i][j], WEIGHTS_FINAL[i][j], "\n")
            }
        }
    }
}

# Test computations of output_layer deltas
test_calc_delta_output_layer<-function() {
    nn <- get_nn_for_test()
    expected = c(1.0)
    output_layer = c(0.1)
    for (i in 1:length(nn@output_layer@values)) {
        nn@output_layer@values[i] = output_layer[i]
    }
    res <- calc_delta_output(expected, nn)
    DELTA_FINAL = c(0.08100000000000002)
    for (i in 1:length(DELTA_FINAL)) {
        if (res[i] == DELTA_FINAL[i]) {
            cat(".")
        } else {
            print("Error: test_calc_delta_output_layer")
            cat(res[i], DELTA_FINAL[i], "\n")
        }
    }
}

# Test computations of deltas though backpropagation
test_calc_deltas<-function() {
    nn <- get_nn_for_test()
    
    source_layer = c(0.1)
    delta_source = c(0.2)
    target_layer = c(0.3, 0.4)
    source_weights =  array(c(0.5, 0.6, NA, NA), dim=c(2,2))
    
    #source = nn.output_layer
    nn@output_layer@values = source_layer
    nn@output_layer@deltas = delta_source
    nn@output_layer@weights = source_weights
    
    nn@hidden_layer@values = target_layer

    res <- calc_deltas(nn@output_layer, nn@hidden_layer)

    DELTA_FINAL = c(0.021, 0.0288)
    for (i in length(DELTA_FINAL)) {
        if (res[i] == DELTA_FINAL[i]) {
            cat(".")
        } else {
            print("Error: test_calc_deltas")
            cat(res[i], DELTA_FINAL[i], "\n")
        }
    }
}

do_tests<-function() {
    test_calc_deltas()
    test_calc_delta_output_layer()
    test_update_weights()
    test_activation_function()
    print("done")
}
