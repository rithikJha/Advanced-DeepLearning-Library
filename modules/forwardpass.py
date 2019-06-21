import numpy as np
import modules.activations as act


def linear_forward(A_prev,W,b):
    m=A_prev.shape[1]
    Z=np.dot(W,A_prev)+b
    linear_cache =(A_prev,W,b)   
    return Z,linear_cache


def linear_activation_forward(A_prev,W ,b,activation):

    if activation.lower()=="sigmoid":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = act.sigmoid(Z)

    if activation.lower()=="relu":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = act.relu(Z)

    if activation.lower()=="tanh":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = act.tanh(Z)

    if activation.lower()=="linear":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = act.linear(Z)

    cache = (linear_cache,activation_cache)
    return A,cache