import numpy as np 

def velocity_initializer(parameters):
    v={}
    L=len(parameters)//2

    for l in range(L):
        v["dW"+str(l+1)]=np.zeros_like(parameters["W"+str(l+1)])
        v["db"+str(l+1)]=np.zeros_like(parameters["b"+str(l+1)])
    return v


def speed_initializer(parameters):
    s={}
    L=len(parameters)//2

    for l in range(L):
        s["dW"+str(l+1)]=np.zeros_like(parameters["W"+str(l+1)])
        s["db"+str(l+1)]=np.zeros_like(parameters["b"+str(l+1)])
    return s    


def update_parameters_momentum(grads,v,beta,t):
    v = beta*v+(1-beta)*grads
    
    return v

def update_parameters_RMSprop(grads,s,beta,t):
    eps = 1e-8
    s = beta*s+(1-beta)*np.square(grads)
      
    return s


def update_parameters(parameters,direction,learning_rate):
    parameters = parameters- learning_rate * direction
    return parameters


def update_parameters_with_optimizer(parameters,grads,v,s,t,learning_rate,beta1,beta2,optimizer):
    L=len(parameters)//2
    direction={}
    eps=1e-8
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {} 
    for l in range(L):
        
        if optimizer.lower()=="momentum":
            v["dW"+str(l+1)] = update_parameters_momentum(grads["dW"+str(l+1)],v["dW"+str(l+1)],beta1,t)
            v["db"+str(l+1)] = update_parameters_momentum(grads["db"+str(l+1)],v["db"+str(l+1)],beta1,t)
            direction = v

        if optimizer.lower()=="":
            direction=grads

        if optimizer.lower() == "adam":
            v["dW" + str(l+1)] = update_parameters_momentum(grads["dW"+str(l+1)],v["dW"+str(l+1)],beta1,t)
            v["db" + str(l+1)] = update_parameters_momentum(grads["db"+str(l+1)],v["db"+str(l+1)],beta1,t)
            v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1-beta1**t)
            v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1-beta1**t)
            s["dW" + str(l+1)] = update_parameters_RMSprop(grads["dW"+str(l+1)],s["dW"+str(l+1)],beta2,t)
            s["db" + str(l+1)] = update_parameters_RMSprop(grads["db"+str(l+1)],s["db"+str(l+1)],beta2,t)
            s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]/(1-beta2**t)
            s_corrected["db" + str(l+1)] = s["db" + str(l+1)]/(1-beta2**t)
            direction["dW"+str(l+1)]=(v_corrected["dW" + str(l+1)]/(np.sqrt(s_corrected["dW" + str(l+1)])+eps))
            direction["db"+str(l+1)] =(v_corrected["db" + str(l+1)]/(np.sqrt(s_corrected["db" + str(l+1)])+eps))

        if optimizer.lower() == "rmsprop":
            s["dW" + str(l+1)] = update_parameters_RMSprop(grads["dW"+str(l+1)],s["dW"+str(l+1)],beta2,t)
            s["db" + str(l+1)] = update_parameters_RMSprop(grads["db"+str(l+1)],s["db"+str(l+1)],beta2,t)
            direction["dW"+str(l+1)]=(grads["dW" + str(l+1)]/(np.sqrt(s["dW" + str(l+1)])+eps))
            direction["db"+str(l+1)] =(grads["db" + str(l+1)]/(np.sqrt(s["db" + str(l+1)])+eps))          

        parameters["W"+str(l+1)]=update_parameters(parameters["W"+str(l+1)],direction["dW"+str(l+1)],learning_rate)
        parameters["b"+str(l+1)]=update_parameters(parameters["b"+str(l+1)],direction["db"+str(l+1)],learning_rate)
        
    return parameters,v,s




    