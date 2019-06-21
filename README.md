# DeepLearning-Library-From-Scratch
Deep Learning library based of deeplearning.ai course (Coursera)

Let us look at all the hyper-parameters and their functionality:

1) layers_dims      -       contains dimensions of each layer (including input(x) and output(y))
                            specification - a) size - input + hidden + output (=hidden + 2)
                                            b) type - list [X,hidden1,hidden2,...,Y]

2) activation_list  -       contains list of activations for each input layer
                            specification - a) size   - layers_dims - 1             # because output/prediction layer needs no activation further
                                            b) type   - list ["g(X)","g(hidden1)","g(hidden2)",...]  # g(a) means activation for a
                                                                                                     # contains list of strings
                                            c) values - "relu","sigmoid","tanh","softmax"(untested),"linear"(buggy) 

3) cost_cache       -       contains name of the cost function to be used
                            specification - a) type   - string
                                            b) values - "cel"(CrossEntropyLoss),"msel"(MSELoss),"softmax"(untested,Cost for softmax activation)   

4) learning_rate    -       most important parameter (self-explanatory)   
                            specification - a) type   - int

5) mini_batch_size  -       Size of mini-batch for mini-batch-gradient-descent (self-explanatory)   
                            specification - a) type   - int

6) num_epochs    -          number of forward and backward passes you want to give to reduce cost (self-explanatory)   
                            specification - a) type   - int

7) print_cost    -          important, if you want to witness that your cost is decreasing    
                            specification - a) type   - boolean

8) beta1         -          hyper-parameter of momentum-optimization and adam-optimization     
                            specification - a) type   - int

9) beta2         -          hyper-parameter of RMSprop-optimization and adam-optimization   
                            specification - a) type   - int

10) A_type        -          Different types of activations requires different types of weight initialization to tackle the problem of vanishing and exploding gradient 
                            specification - a) type   - string
                                            b) values - "relu" , "xavier" , ""

11) optimizer     -          Type of optimization you want to use !    
                            specification - a) type   - string
                                            b) values - "adam" , "momentum" , "rmsprop" , ""


12) Reg_type      -          Type of Regularization you want to use !    
                            specification - a) type   - string
                                            b) values - "l2" , "l1" , ""

11) lambd         -          hyper-parameter for Regularization effect    
                            specification - a) type   - int


In this Library , it is compulsory to assign - X , Y , layers_dims ! 
Others default settings are stated below (however you can customize it by passing the values in the arguments while calling train.model() ) - 
1) activation_list - ["relu","relu",...,"sigmoid]
2) cost_cache - "cel"
3) learning_rate - 0.0007
4) mini_batch_size - m (number of training example, it can also be achieved by setting the value to zero if you are passing in argument)
5) num_epochs - 10000
6) print_cost -True
7) beta1 - 0.9
8) beta2 - 0.999
9) A_type - ""
10) optimizer - "" (Normal gradient descent)
11) Reg_type - "" (No regularization
12) lambd - 0.1
The above 12 things are customizable but if you don't want to, you can opt-out of passing it as argument 


