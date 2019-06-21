import modules.isthara_versatile as ist
import matplotlib.pyplot as plt
import modules.initialize as ini  
import numpy as np
import math




def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k * mini_batch_size:(k + 1) * mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        end = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches



def model(X, Y, layers_dims,activation_list,cost_cache="cel",learning_rate=0.0007,  
            mini_batch_size=0,num_epochs=10000,print_cost=True,
            beta1=0.9, beta2=0.999,A_type="",optimizer="",Reg_type="",lambd=0.1):  
            


    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours

    if mini_batch_size == 0:
        mini_batch_size=X.shape[1]
    
    
    # Initialize parameters
    parameters = ist.intialize_parameters(layers_dims,A_type)
    v={}
    s={}

    if optimizer.lower()=="adam":
        v=ini.velocity_initializer(parameters)
        s=ini.speed_initializer(parameters)   

    if optimizer.lower()=="rmsprop":
        s=ini.speed_initializer(parameters)

    if optimizer.lower()=="momentum":
        v=ini.velocity_initializer(parameters)

    if mini_batch_size != X.shape[1]:
        for i in range(num_epochs):                                                                                                         # Optimization loop
            # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
            seed = seed + 1
            minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch                                                                                      # Select a minibatch
                AL, caches = ist.linear_deep_forward(minibatch_X, parameters,activation_list)                                               # Forward propagation
                cost = ist.compute_cost(AL, minibatch_Y,cost_cache,lambd,parameters,Reg_type)                                               # Compute cost
                grads = ist.linear_deep_backward(AL, minibatch_Y, caches,activation_list,cost_cache,parameters,lambd,Reg_type)              # Backward propagation
                t = t + 1                                                                                                                   # Adam counter
                parameters, v, s = ini.update_parameters_with_optimizer(parameters, grads, v, s, t,
                                                                    learning_rate, beta1, beta2,optimizer)                                  #Update Parameter

            # Print the cost every 1000 epoch
            if print_cost and i % 1000 == 0:
                print("Cost after epoch %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)


    if mini_batch_size == X.shape[1]:
        for i in range(num_epochs):
            AL,caches = ist.linear_deep_forward(X,parameters,activation_list)                                              # Forward Propagation
            cost = ist.compute_cost(AL,Y,cost_cache,lambd,parameters,Reg_type)                                             # Compute Cost
            grads = ist.linear_deep_backward(AL,Y,caches,activation_list,cost_cache,parameters,lambd,Reg_type)             # Backward Propagation
            t = t + 1
            parameters, v, s = ini.update_parameters_with_optimizer(parameters, grads, v, s,
                                                                   t, learning_rate, beta1, beta2,optimizer)               # Update Parameters

            if print_cost and i % 1000 == 0:
                print("Cost after epoch %i: %f" % (i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)


    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


def predict(X, y, parameters,activation_list):
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)

    
    # Forward propagation
    a3, caches = ist.linear_deep_forward(X, parameters,activation_list)
    
    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
    return p


