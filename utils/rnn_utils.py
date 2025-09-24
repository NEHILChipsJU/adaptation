import numpy as np
from scipy.sparse import random 
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# fail=np.array([10,250], dtype=int)
n=3

#container that randomly initializes W and Win
def rnn_params(
    rnn_size,
    input_size,
    output_size,
    input_scaling,
    spectral_radius,
    a_dt,
    bias_scaling,
    sparsity,
    seed=1235
    
):
    """
    Initializes the parameters for a simple RNN model.

    Args:
    - rnn_size (int): The number of hidden units in the RNN.
    - input_size (int): The number of input features.
    - output_size (int): The number of output features.
    - input_scaling (float): Scaling factor for the input weights.
    - spectral_radius (float): Desired spectral radius of the recurrent weight matrix.
    - a_dt (float): Time step size.
    - bias_scaling (float, optional): Scaling factor for the bias terms. Defaults to 0.8.
    - seed (int, optional): Seed for the random number generator. Defaults to 1235.
    - Sparsity (int or None): If sparsity is needed you can give a value, if not None
    
    Returns:
    - params (dict): A dictionary containing the initialized parameters.
    """

    prng = np.random.default_rng(seed)
    def normal_data(n):
        # return prng.normal(loc=0,scale=1.0,size=n)
        return prng.normal(size=n)
    #crea la matriz W
    def rnn_ini(shape, spectral_radius,sparsity): #matrix dimension and desired spectral radius
        if sparsity is not None and sparsity!=1:
            w=random(shape[0], shape[1], density=sparsity, data_rvs=normal_data, format='coo',random_state=prng.integers(1e9)) #if we want to give sparsity to the matrix
            w=w.toarray()
        else:
            w = prng.normal(size=shape)
            # w = prng.normal(loc=0,scale=1.0,size=shape) #internal weights, generates a matrix with Gaussian distribution (values range from -1 to 1)
        current_spectral_radius = max(abs(np.linalg.eig(w)[0])) # calcula el radio expectral de la matriz w aleatoria
        w *= spectral_radius / current_spectral_radius # adjusts W by scaling all its values so that its new spectral radius equals the specified spectral_radius
                                                        # this controls the recurrent dynamics (prevents exploding or vanishing activations in the RNN)
        return w
    
    params = dict(
        win=prng.normal(loc=0,scale=1.0,size=(rnn_size, input_size)) * input_scaling, #input weight matrix with range [-a, a]
        w=rnn_ini((rnn_size, rnn_size), spectral_radius,sparsity), #calls the function to create the matrix
        bias=prng.normal(size=(rnn_size,)) * bias_scaling, #the bias vector for hidden layers, with dimension Nx1 where N is the number of internal neurons
        wout=prng.normal(size=(output_size, rnn_size)), #output weight matrix 1xN, 1 output neurone
        bias_out=prng.normal(size=(output_size,)) * bias_scaling, #b fot output layer
        a_dt=a_dt * np.ones(rnn_size), 
        x_ini=0.1 * prng.normal(size=(rnn_size)), # random inital hidden state
    )

    return  params







#paper first equation
def forward_rnn(params, ut,x_init=None, autonomous=False,conceptor=None): #autonomous mode False by default
    """
    Forward pass of a recurrent neural network (RNN) .

    Args:   
    - params (dict): dictionary containing the RNN parameters (weights and biases).
    - ut (ndarray): input to the RNN.
    - x_init (ndarray, optional): initial state of the RNN. Defaults to None.
    - autonomous (boolean): True or False if we want to use this mode or not
    - conceptor (array): The conceptor we want to use or None
    

    Returns:
    - X (matriz): hidden satate for all the time series
    
    
    use params_trained for every case that you use this function after training the model
    """
    if x_init is None:
        x = params["x_ini"]
    else:
        x = x_init
    x = np.ravel(x)      
    T=len(ut)
    N=params['w'].shape[0]
    # Creating the container for the state matrix
    X = np.zeros((T, N))
    if conceptor is None: # si es el primer paso, antes del entrenamiento
        conceptor = np.eye(x.shape[0]) #hace un conceptor identidad
    else:
        conceptor=conceptor
    # temporal loop
   
    for t_idx in range(T):#iterating through the time vector
          
        u_t = (
            ut[t_idx] if not autonomous else np.dot(params["wout"], x) + params["bias_out"]
            )
        #The part inside the tanh
        dentro = params["w"] @ x \
            + params["win"] @ u_t \
            + params["bias"]
    
        # Updating 'leaky tanh', element-wise multiplication
        x = conceptor @ ((1 - params["a_dt"]) * x \
             + params["a_dt"] * np.tanh(dentro))
        x=np.ravel(x)
        # Storing the hidden state
        X[t_idx] = x
        
    return X



#paper first equation
def forward_rnn_deg(params, ut,x_init=None, autonomous=False,conceptor=None,deg=0,seed=42): #autonomous mode False by default
    """
    Forward pass of a recurrent neural network (RNN) .

    Args:   
    - params (dict): dictionary containing the RNN parameters (weights and biases).
    - ut (ndarray): input to the RNN.
    - x_init (ndarray, optional): initial state of the RNN. Defaults to None.
    - autonomous (boolean): True or False if we want to use this mode or not
    - conceptor (array): The conceptor we want to use or None
    - deg (float): % of neuron degradation
    - seed: seed for the random degradation

    Returns:
    - X (matriz): hidden satate for all the time series
    
    
    use params_trained for every case that you use this function after training the model
    """
    if x_init is None:
        x = params["x_ini"]
    else:
        x = x_init
    x = np.ravel(x)    
    T=len(ut)
    N=params['w'].shape[0]
    #generating the index fot the fail nodes
    _,fail=degradation(x,deg,seed)
    # Creating the container for the state matrix
    X = np.zeros((T, N))
    if conceptor is None: # si es el primer paso, antes del entrenamiento
        conceptor = np.eye(x.shape[0]) #hace un conceptor identidad
    else:
        conceptor=conceptor
    # temporal loop
   
    for t_idx in range(T):#iterating through the time vector
          
        u_t = (
            ut[t_idx] if not autonomous else np.dot(params["wout"], x) + params["bias_out"]
            )
        #introducing the degradation manually
        # if t_idx> int(T/n):
        #     x[fail]=0
        #The part inside the tanh
        d = params["w"] @ x \
            + params["win"] @ u_t \
            + params["bias"]
    
        # Updating 'leaky tanh', element-wise multiplication
        x = ((1 - params["a_dt"]) * x \
             + params["a_dt"] * np.tanh(d))
        
            
        x=conceptor @ x
        if t_idx> int(T/n):
            x[fail]=0
        # Storing the hidden state
        X[t_idx] = x
        #introducing the degradation manually
        
    return X


# eq 3 paper
def compute_conceptor(X, aperture):
    """
    Computes the conceptor matrix for a given input matrix X and an aperture value.

    Arg:
    - X (numpy.ndarray): Input matrix of shape (n_samples, n_features). (t,N)
    - aperture (float): Aperture value used to compute the conceptor matrix.
   
    Returns:
    - C (ndarray) Conceptor matrix of shape (n_features, n_features). (N,N)
    """
    R = np.dot(X.T, X) / X.shape[0]
    
    C = np.dot(R, np.linalg.inv(R + aperture ** (-2) * np.eye(R.shape[0])))
    return C



#getting Wout with ridge regression of Scikit-Learn

def ridge(beta,X,Y_target,step,params):
    """
    Trainning the model and visualize the results for the input dataset
    
    Arg:
    - beta:(float) ridge coefficient
    - X (array): hidden state matrix (T,N)
    - Y_target(array): target outpot values (T,1)
    - step: (int) number of steps the model has tu predict
    - params (dict): the ESN params
    
    Returns:
    - params_trained (dict): same ESN params but with the trained Wout and bias_out
    - mse (float): mean squared error between the prediction and the real signal
    
    """
    #Generating the model with beta
    ridge_model = Ridge(alpha=beta, fit_intercept=True) #fit_intercept=True if we want to train bias_out
    
    #Training the model with X and Y_target
    ridge_model.fit(X, Y_target)
    
    #getting Wout (1,N)
    W_out = ridge_model.coef_
    bias_out = ridge_model.intercept_
    
    #Predict Y with this Wout    
    Y_pred = X @ W_out.T + bias_out
    
    #computing the mse
    mse = np.mean((Y_pred[:-step] - Y_target[:-step])**2)
    
    #updating params
    params['wout']=W_out
    params['bias_out']=bias_out
    params_trained=params
    
    return params_trained, mse






def trained_model_new(X,ut,yt,params_trained,washout,autonomous,conceptor,label=None):
    """
    Using the trained model in order to predict a new input
    
    Args:
    - X (ndarray): internal state that we want to plot
    - ut (ndarray): input to the RNN.
    - yt (ndarray): real output of the RNN.
    - params_trained (dict): dictionary containing the RNN parameters trained (weights and biases).
    - washout (int): number of steps for the washout
    - autonomous (boolean): True or False if we want to use this mode or not
    - conceptor (array): The conceptor we want to use or None
    - label (string): if we want to give an especific title to the plot
    
    Returns:
    - plot
    
    """
    
    #deleting the initial states
    X = X[washout:]
    yt = yt[washout:]
    #Predict Y with this Wout    
    Y_pred = X @ params_trained['wout'].T + params_trained['bias_out']
    #computing the mse
    
    #showing the predictiosn versus the real data
    plt.figure()
    plt.plot(yt, label="Real (Target)")
    
    if label is not None:
        plt.title(label)
    else:
        if autonomous is False:
            plt.title("1-step ahead prediction on new sine wave")
            mse_new = np.mean((Y_pred - yt)**2)
            return mse_new
        else:
            plt.title("1-step ahead, Autonomous")
        if autonomous is True and conceptor is not None:
            plt.title("1-step ahead, Autonomous with conceptor")
    plt.xlabel('k')
    plt.plot(Y_pred, label="Predicted (ESN)")
    
    plt.legend()
    plt.show()
    
    




def forward_rnn_CL(nu,a,beta,params, ut,C_init,C_target,C1,C2,x_init=None,autonomous=False): #autonomous mode False by default
    """
    Forward pass of a recurrent neural network (RNN) .With Conceptor control loop

    Args:
    - nu: learning rate  
    - a: aperture
    - beta: control gain 
    - params (dict): dictionary containing the RNN parameters (weights and biases).
    - ut (ndarray): input signal. For lambda scan ut->t_scan 
    - x_init (ndarray, optional): initial state of the RNN. Defaults to None.
    - C_init: initial conceptor for the autoconceptor eq C(0).
    - C_target : Traget conceptor for the CCL logic
    - C1, C2: sine(T1), sine(T2) input conceptor. For the lambda scan
    - Autonomous: True or False if the autonomous mode is used or not

    Returns:
    - X (matriz): hidden satate for all the time series
    - C_aut: last autoconceptor
    
    
    use params_trained for every case that you use this function anter training the model

    """
    if x_init is None:
        x = params["x_ini"]
    else:
        x = x_init
    x = np.ravel(x)  
      
    T=len(ut)
    N=params['w'].shape[0]
    # Creating the container for the state matrix
    X = np.zeros((T, N))
    #Seting the initial Cs
    C_aut=C_init
    C_t=C_target
    #setting the lambda scan
    lamda = np.linspace(0, 1, T)
    # temporal loop
    for t_idx in range(T):#iterating through the time vector
        
        #if we dont have a target, the target is the lambda scan C
        #lambda scan
        if C_target is None:
            C_t=lamda[t_idx]*C1+(1-lamda[t_idx])*C2
        #autoconceptor
        # C_aut=C_aut+nu*((x-C_aut @ x) @ x.T-(a**(-2))*C_aut)
        C_aut = C_aut + nu*(np.outer(x - C_aut @ x, x) - (a**(-2)*C_aut))

        #adapting the conceptor to the target   
        C_adapt= C_aut-beta*(C_aut-C_t)
         
        u_t =  (
            ut[t_idx] if not autonomous else np.dot(params["wout"], x) + params["bias_out"]
            )
        #The part inside the tanh
        d = params["w"] @ x \
            + params["win"] @ u_t \
            + params["bias"]
    
        # Updating 'leaky tanh', element-wise multiplication
        # leaky tanh + conceptor
        x = (1 - params["a_dt"]) * x + params["a_dt"] * np.tanh(d)
        x = C_adapt @ x
         
        
        # Storing the hidden state
        X[t_idx] = x
        
        
    return X,C_adapt


def forward_rnn_CL_deg(nu,a,beta,params, ut,C_init,C_target,C1,C2,x_init=None,autonomous=False,deg=0,seed=42): #autonomous mode False by default
    """
    Forward pass of a recurrent neural network (RNN) .With Conceptor control loop

    Args:
    - nu: learning rate  
    - a: aperture
    - beta: control gain 
    - params (dict): dictionary containing the RNN parameters (weights and biases).
    - ut (ndarray): input signal. For lambda scan ut->t_scan 
    - x_init (ndarray, optional): initial state of the RNN. Defaults to None.
    - C_init: initial conceptor for the autoconceptor eq C(0).
    - C_target : Traget conceptor for the CCL logic
    - C1, C2: sine(T1), sine(T2) input conceptor. For the lambda scan
    - Autonomous: True or False if the autonomous mode is used or not
    - deg (float): % of neuron degradation
    - seed: seed for the random degradation
    Returns:
    - X (matriz): hidden satate for all the time series
    - C_aut: last autoconceptor
    
    
    use params_trained for every case that you use this function anter training the model

    """
    if x_init is None:
        x = params["x_ini"]
    else:
        x = x_init
    x = np.ravel(x)  
          
    T=len(ut)
    N=params['w'].shape[0]
    # Creating the container for the state matrix
    X = np.zeros((T, N))
    #Seting the initial Cs
    C_aut=C_init
    C_t=C_target
    #setting the lambda scan
    lamda = np.linspace(0, 1, T)
    #generating the index fot the fail nodes
    _,fail=degradation(x,deg,seed)
    # temporal loop
    for t_idx in range(T):#iterating through the time vector
        
        #if we dont have a target, the target is the lambda scan C
        #lambda scan
        if C_target is None:
            C_t=lamda[t_idx]*C1+(1-lamda[t_idx])*C2
            
        #autoconceptor
        # C_aut=C_aut+nu*((x-C_aut @ x) @ x.T-(a**(-2))*C_aut)
        C_aut = C_aut + nu*(np.outer(x - C_aut @ x, x) - (a**(-2)*C_aut))

        #adapting the conceptor to the target   
        C_adapt= C_aut-beta*(C_aut-C_t)
         
        u_t =  (
            ut[t_idx] if not autonomous else np.dot(params["wout"], x) + params["bias_out"]
            )
        
        #introducing manual degradation
        # if t_idx> int(T/n):
        #     x[fail]=0
            
        #The part inside the tanh
        d = params["w"] @ x \
            + params["win"] @ u_t \
            + params["bias"]
        
        
        
        # Updating 'leaky tanh', element-wise multiplication
        # leaky tanh + conceptor
        x = (1 - params["a_dt"]) * x + params["a_dt"] * np.tanh(d)
        
        #introducing manual degradation
        # if t_idx> int(T/n):
        #     x[fail]=0
            
        x = C_adapt @ x    
        #introducing manual degradation
        if t_idx> int(T/n):
            x[fail]=0
        # Storing the hidden state
        X[t_idx] = x

    return X, C_adapt

def forward_rnn_interp(params, t,C1,Cs,x_init=None): #autonomous mode True by default
    """
    Forward pass of a recurrent neural network (RNN) . If we want to show Paper's Fig3 we need to vary lambda and t at the same time

    Args:
    - N: rnn size    
    - params (dict): dictionary containing the RNN parameters (weights and biases).
    - t (ndarray): Number of t steps
    - idx (int): index of the RNN to use (in case multiple RNNs are stored in params).
    - x_init (ndarray, optional): initial state of the RNN. Defaults to None.
    

    Returns:
    - X (matriz): hidden satate for all the time series
    
    
    use params_trained for every case that you use this function anter training the model

    """
    if x_init is None:
        x_init = params["x_ini"]
        
    x = np.ravel(x_init)  # <-- always 1D   
    T=len(t)
    N=params['w'].shape[0]
    # Creating the container for the state matrix
    X = np.zeros((T, N))
    #setting the lambda scan
    lamda = np.linspace(0, 1, T)
    # temporal loop
   
    for t_idx in range(T):#iterating through the time vector
        
        #lambda scan
        
        C_lam=lamda[t_idx]*C1+(1-lamda[t_idx])*Cs
           
                 
        u_t = (
            np.dot(params["wout"], x) + params["bias_out"]
            )
        #The part inside the tanh
        d = params["w"] @ x \
            + params["win"] @ u_t \
            + params["bias"]
    
        # Updating 'leaky tanh', element-wise multiplication
        # leaky tanh + conceptor
        x = (1 - params["a_dt"]) * x + params["a_dt"] * np.tanh(d)
        x = C_lam @ x
        x = np.ravel(x)  
    
        # Storing the hidden state
        X[t_idx] = x
        
    return X


def forward_rnn_mix_deg(nu,a,beta,params, ut,C_init,C_target,C1,C2,x_init=None,autonomous=False,deg=0,seed=42): #autonomous mode False by default
    """
    Forward pass of a recurrent neural network (RNN) .With Conceptor control loop

    Args:
    - nu: learning rate  
    - a: aperture
    - beta: control gain 
    - params (dict): dictionary containing the RNN parameters (weights and biases).
    - ut (ndarray): input signal. For lambda scan ut->t_scan 
    - x_init (ndarray, optional): initial state of the RNN. Defaults to None.
    - C_init: initial conceptor for the autoconceptor eq C(0).
    - C_target : Traget conceptor for the CCL logic
    - C1, C2: sine(T1), sine(T2) input conceptor. For the lambda scan
    - Autonomous: True or False if the autonomous mode is used or not
    - deg (float): % of neuron degradation
    - seed: seed for the random degradation

    Returns:
    - X (matriz): hidden satate for all the time series
    - C_aut: last autoconceptor
    
    
    use params_trained for every case that you use this function anter training the model

    """
    if x_init is None:
        x = params["x_ini"]
    else:
        x = x_init
    x = np.ravel(x)  
          
    T=len(ut)
    N=params['w'].shape[0]
    # Creating the container for the state matrix
    X = np.zeros((T, N))
    #Seting the initial Cs
    C_adapt=C_init
    C_t=C_target
    #setting the lambda scan
    lamda = np.linspace(0, 1, T)
    #generating the index fot the fail nodes
    _,fail=degradation(x,deg,seed)
    # temporal loop
    for t_idx in range(T):#iterating through the time vector
        
        #if we dont have a target, the target is the lambda scan C
        #lambda scan
        if C_target is None:
            C_t=lamda[t_idx]*C1+(1-lamda[t_idx])*C2
            
        
        #autoconceptor
        # C_aut=C_aut+nu*((x-C_aut @ x) @ x.T-(a**(-2))*C_aut)
        C_adapt = C_adapt + nu*(np.outer(x - C_adapt @ x, x) - (a**(-2)*C_adapt))-beta*(C_adapt-C_t)

        
         
        u_t =  (
            ut[t_idx] if not autonomous else np.dot(params["wout"], x) + params["bias_out"]
            )
        #The part inside the tanh
        d = params["w"] @ x \
            + params["win"] @ u_t \
            + params["bias"]
    
        # Updating 'leaky tanh', element-wise multiplication
        # leaky tanh + conceptor
        x = (1 - params["a_dt"]) * x + params["a_dt"] * np.tanh(d)
        x = C_adapt @ x
        #introducing manual degradation
        if t_idx> int(T/n):
            x[fail]=0
            
        # Storing the hidden state
        X[t_idx] = x

    return X, C_adapt



def forward_rnn_mix(nu,a,beta,params, ut,C_init,C_target,C1,C2,x_init=None,autonomous=False): #autonomous mode False by default
    """
    Forward pass of a recurrent neural network (RNN) .With Conceptor control loop

    Args:
    - nu: learning rate  
    - a: aperture
    - beta: control gain 
    - params (dict): dictionary containing the RNN parameters (weights and biases).
    - ut (ndarray): input signal. For lambda scan ut->t_scan 
    - x_init (ndarray, optional): initial state of the RNN. Defaults to None.
    - C_init: initial conceptor for the autoconceptor eq C(0).
    - C_target : Traget conceptor for the CCL logic
    - C1, C2: sine(T1), sine(T2) input conceptor. For the lambda scan
    - Autonomous: True or False if the autonomous mode is used or not

    Returns:
    - X (matriz): hidden satate for all the time series
    - C_aut: last autoconceptor
    
    
    use params_trained for every case that you use this function anter training the model

    """
    if x_init is None:
        x = params["x_ini"]
    else:
        x = x_init
    x = np.ravel(x)  
          
    T=len(ut)
    N=params['w'].shape[0]
    # Creating the container for the state matrix
    X = np.zeros((T, N))
    #Seting the initial Cs
    C_adapt=C_init
    C_t=C_target
    #setting the lambda scan
    lamda = np.linspace(0, 1, T)
    # temporal loop
    for t_idx in range(T):#iterating through the time vector
        
        #if we dont have a target, the target is the lambda scan C
        #lambda scan
        if C_target is None:
            C_t=lamda[t_idx]*C1+(1-lamda[t_idx])*C2
            
        
        #autoconceptor
        # C_aut=C_aut+nu*((x-C_aut @ x) @ x.T-(a**(-2))*C_aut)
        C_adapt = C_adapt + nu*(np.outer(x - C_adapt @ x, x) - (a**(-2)*C_adapt))-beta*(C_adapt-C_t)

        
         
        u_t =  (
            ut[t_idx] if not autonomous else np.dot(params["wout"], x) + params["bias_out"]
            )
        #The part inside the tanh
        d = params["w"] @ x \
            + params["win"] @ u_t \
            + params["bias"]
    
        # Updating 'leaky tanh', element-wise multiplication
        # leaky tanh + conceptor
        x = (1 - params["a_dt"]) * x + params["a_dt"] * np.tanh(d)
        x = C_adapt @ x
       
            
        # Storing the hidden state
        X[t_idx] = x

    return X, C_adapt



def degradation(matrix, zero_percentage,seed):
    """
    Randomly adds zeros to a matrix.

    Args:
    - matrix(NumPy array): The matrix to modify.
    - zero_percentage: The percentage of elements that will be turned into zeros (0–100).
      
    Results:
    - matrix_with_zeros (NumPy array): The final matrix with the zeros 
    - index (NumPy array): random indexs selected for the ceros   
    """
    
    #how many elements should be zero:
    num_elements = matrix.size
    num_zeros = int(num_elements * (zero_percentage / 100))

    # Get random indices to replace with zeros
    rng = np.random.default_rng(seed)
    index = rng.choice(num_elements, num_zeros, replace=False)

    # Create a mask for the indices that will become 0
    mask = np.zeros(num_elements, dtype=bool)
    mask[index] = True

    # Flatten the matrix into a 1D array
    flat_matrix = matrix.flatten()

    # Replace the selected elements with 0
    flat_matrix[mask] = 0

    # Reshape back to the original matrix shape
    matrix_with_zeros = flat_matrix.reshape(matrix.shape)

    return matrix_with_zeros, index




