import numpy as np 
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import math
import modules.trainMeDeep as train 


def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=3000, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=1000, noise=.05)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

def load_dataset2():
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
    '''# Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);'''
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    
    return train_X, train_Y




train_X, train_Y = load_dataset2()

layers_dims = [train_X.shape[0], 5, 2, 1]
activation_list = ["relu","relu","sigmoid"]
cost_cache="cel"





parameter = train.model(train_X, train_Y, layers_dims,activation_list,cost_cache,
                    learning_rate=0.0007,mini_batch_size=64,num_epochs=10000,print_cost=True,
                    beta1=0.9, beta2=0.999,A_type="relu",optimizer="adam",Reg_type="",lambd=0.1)

predictions = train.predict(train_X, train_Y, parameter,activation_list)
