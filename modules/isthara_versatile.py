import modules.activations as act
import modules.forwardpass as fwd  
import modules.backpass as back
import numpy as np

def intialize_parameters(layers_dims,A_type) -> dict:
    np.random.seed(3)
    L =len(layers_dims)
    parameters={}
    for l in range(1,L):
        if A_type.lower()=="relu":
            parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
            parameters["b"+str(l)]=np.zeros((layers_dims[l],1))

        elif A_type.lower()=="xavier":
            parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/(layers_dims[l-1]*layers_dims[l]))
            parameters["b"+str(l)]=np.zeros((layers_dims[l],1))       

        elif A_type.lower()=="":
            parameters["W"+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(1/layers_dims[l-1])
            parameters["b"+str(l)]=np.zeros((layers_dims[l],1)) 

    return parameters



def linear_deep_forward(X,parameters,activation_list):

    L=len(parameters)//2
    caches=[]
    A_prev=X

    for l in range(1,L+1):
            A,cache=fwd.linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],activation_list[l-1])
            caches.append(cache)
            A_prev=A

    AL = A_prev
    return AL,caches

def compute_cost(AL,Y,cost_cache,lambd,parameters,Reg_type):
    m=Y.shape[1]
    eps=1e-8
    reg_cost=back.regularization_cost(lambd,parameters,m,Reg_type)
    if cost_cache.lower()=="cel":
        cost = (-1/m)*(np.dot(Y,np.log(AL+eps).T)+np.dot(1-Y,np.log(1-AL+eps).T)) + reg_cost
    if cost_cache.lower()=="msel":
        cost = np.square(AL-Y).mean()/2 +reg_cost
    if cost_cache.lower() == "softmax":
        cost = -1*np.sum(np.dot(Y,np.log(AL).T))/m + reg_cost

    cost = np.squeeze(cost)
    return cost


def linear_deep_backward(AL,Y,caches,activation_list,cost_cache,parameters,lambd,Reg_type):
    L=len(caches)
    grads={}
    m=AL.shape[1]
    eps=1e-8
    zeta = AL-Y
    Y=Y.reshape(AL.shape)
    if cost_cache.lower()=="cel":
        dAL = -np.divide(Y,AL+eps)+np.divide(1-Y,1-AL+eps)
    if cost_cache.lower()=="msel":
        dAL=zeta
    dA=dAL
    for l in reversed(range(L)):
        current_cache = caches[l]
        reg_term =back.regularization_term(lambd,parameters["W"+str(l+1)],m,Reg_type)
        grads["dA"+str(l)],grads["dW"+str(l+1)],grads["db"+str(l+1)]=back.linear_activation_backward(dA,current_cache,activation_list[l],zeta) 
        grads["dW"+str(l+1)] += reg_term
        dA=grads["dA"+str(l)]

    return grads
