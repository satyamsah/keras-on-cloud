# keras-on-cloud



####

    1. instantiate the sequentail layers
        1. sequential class which takes in list of layers for eg: 2 dense layer , 2 activation layer(1 relu and 1 softmax)

#### 
    1. It only needs shape of the first layer in Sequential model.
    2. Subsequent layer infer thier own shape: this is easy 

####
    - layers will have common interface:
       1.layer.getweight()
       2.layer.setweight()
       3.layer.get_config()

    two types of layers:
    - single node
        - single input
        - All layers in Sequential models
    - shared : 
        - multiple input
        - may occur in functional API models
    
####
    - compile our model
      model.compile: ties our model with backend like  Tensorflow theano or CNTK
      - must need to specify optimizer and loss fucntion
      
####
    - train out model:
       -- forward prop: 
            ---input arguments:
               X- input data of sixe(nx,m)
               parmameter: W1,W2(because the data has 2 features) and b1 and, b1 and b2(nh*1) 
            --- compute Z1, that is w2*x+b and then colcualte Z2 that is using passing A1(Z1) which gives activation output
          
      -- compute cost
              logprobs = np.multiply(Y,np.log(A2))+np.multiply((1-Y),np.log(1-A2))
               cost =np.sum(logprobs)*(-1/m)
      
       - back prop:
         -- 
           Implement the backward propagation using the instructions above.
    
                Arguments:
                parameters -- python dictionary containing our parameters 
                cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
                X -- input data of shape (2, number of examples)
                Y -- "true" labels vector of shape (1, number of examples)

                Returns:
                grads -- python dictionary containing your gradients with respect to different parameters
                """
                m = X.shape[1]

                # First, retrieve W1 and W2 from the dictionary "parameters".
                ### START CODE HERE ### (≈ 2 lines of code)
                W1 = parameters["W1"]
                W2 =  parameters["W2"]
                ### END CODE HERE ###

                # Retrieve also A1 and A2 from dictionary "cache".
                ### START CODE HERE ### (≈ 2 lines of code)
                A1 = cache["A1"]
                A2 = cache["A2"]
                ### END CODE HERE ###

                # Backward propagation: calculate dW1, db1, dW2, db2. 
                ### START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
                dZ2 = A2-Y
                dW2 = 1/m*np.dot(dZ2,A1.T)
                db2 = 1/m*np.sum(dZ2,axis=1,keepdims=True)
                #test start
                #print("size 1- np.power(A1, 2)",np.size(1- np.power(A1, 2)))
                #print("type 1- np.power(A1, 2)",type(1- np.power(A1, 2)))
                #print("size (1- np.power(A1, 2))",np.size((1- np.power(A1, 2))))
                #print("type (1- np.power(A1, 2))",type((1- np.power(A1, 2))))
                #print(" 1- np.power(A1, 2)",1- np.power(A1, 2))
                #print("np.dot(W2.T,dZ2)",np.dot(W2.T,dZ2))
                #test end
                dZ1 = np.dot(W2.T,dZ2)*(1- np.power(A1, 2))
                #print("dZ1",dZ1)

                dW1 = 1/m* np.dot(dZ1,X.T)
                db1 = 1/m* np.sum(dZ1,axis=1,keepdims=True)
                ### END CODE HERE ###

                grads = {"dW1": dW1,
                         "db1": db1,
                         "dW2": dW2,
                         "db2": db2}

####
    - save a model:
        -- save to disk 2 different components 
        -- 1. model artichitecture: it contans layers and interconnection netween the layers
        -- 2. model paprmeter: they  are the wieghts and biases of all my interconnections
        -- json or yaml files
        -- model wieghts are saved for hd5 formts
              -- grid format
              -- optimized for high dimensional arrays
              -- can save wights durin/after training
              
        
        
         
      
      
      
       