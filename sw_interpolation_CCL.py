#Import Libraries
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

#Import predefined functions

from utils.rnn_utils import rnn_params
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import forward_rnn_CL
from utils.rnn_utils import ridge
from utils.rnn_utils import trained_model_new
from utils.rnn_utils import compute_conceptor

from utils.utils import T_instant
from utils.utils import visualize_PCA_3D
from utils.utils import visualize_sine_interpolation
from utils.utils import visualize_sine_interpolation_one

#########################################################################

##Random inicialization of the ESN, with the parameters of the paper

########################################################################

spectral_radius=1.6
scaling=1
bias_scaling=1
alpha=0.75 #Leakage
a=200 #Aperture.  
N=256 # Network size 256
nu=1e-6 #Learning Rate
beta=1 #Control gain
washout=200 # steps we wait until the network is stable, in order to show the results
reg=2.5e-5 #regularization parameter in the Ride regression
step=1 # number of steps that the model will predict
sparsity=None
#time steps for the Fig 3 interpolation
n_steps=150000
t_interp=np.arange(0,n_steps-1,1)

#################################################################################

# Setting the different inputs 

################################################################################

#Input Signal
def sin(T,t, A):
    """Sinus wave with a given period T, a time series (t) and an Amplitude (A)"""
    s=A*np.sin((2*np.pi*t)/T)
    return s

#Defining the Temporal Series
dt=0.1
t=np.arange(0, 300,dt)
A=1 #Sinus Amplitude
#  we will use more than one input 
T1=20 #fixed period
T2=[25] #variable period, we want to compare differents T2 with the same T1

#getting the inputs using the difined sine function
data1=sin(T1,t,A)
data1=data1.reshape(-1, 1)
#create shifted input and output, the output is the target
ut_train1 = data1[:-step]               # shape (N-step, 1)
yt_train1 = data1[step:]                # shape (N-step, 1)

#get dimensions
input_size=ut_train1.shape[-1]
output_size=yt_train1.shape[-1]

#################################################################################

#building a dictionary with the parameters that will form the ESN

#################################################################################

#initalize parameters
params=rnn_params(
    N,
    input_size,
    output_size,
    scaling,
    spectral_radius,
    alpha,
    bias_scaling,
    sparsity,
    seed=1235
)

#################################################################################

#Getting sin(T1) internal state matrix and the conceptor

#################################################################################

#obtain matrix X1 (time, N) of internal states for all time points
X1=forward_rnn(params, ut_train1, x_init=None,autonomous=False,conceptor=None)
#Compute model conceptors
C1=compute_conceptor(X1, a)
#Re computing X2
X1_new=forward_rnn(params, ut_train1, X1[-1],False,C1)
# visualize_PCA_3D(X1_new,f' X. T0= {T1}')

###########################################################################

#Loop for the different T2 

##############################################################################

#generating the place where the differents concat will be stored
X_concats = []
yt_targets = []
Cs=[]
C_interps=[]
ut_concats=[]
y_lams=[]

for i in tqdm(range(len(T2))):
    
    #Showing the 2 sinus
    plt.figure()
    plt.plot(t,sin(T1,t,A),label=f"T1 = {T1}")
    plt.plot(t,sin(T2[i],t,A),label=f"T2 = {T2[i]}")
    plt.xlabel('t(s)')
    plt.title("Showing the inputs")
    plt.legend()
    
    #define a variable with the training data and the targets (to predict the last point of the time series)
    data2=sin(T2[i],t,A)
    data2=data2.reshape(-1, 1)
    #create shifted input and output, the output is the target
    ut_train2 = data2[:-step]               # shape (N-step, 1)
    yt_train2 = data2[step:]                # shape (N-step, 1)
    
    #Storing the input and the expected output in a dictionary
    ut_concat=np.concatenate((ut_train1,ut_train2))
    yt_target=np.concatenate((yt_train1,yt_train2))
    ut_concats.append(ut_concat)
    yt_targets.append(yt_target)
    
    #############################################################################
   
    ##Extracting X for each T2
   
    #############################################################################
    
    X2=forward_rnn(params, ut_train2, x_init=None,autonomous=False,conceptor=None)
    
    #Compute T2 conceptors
    C=compute_conceptor(X2, a)
    Cs.append(C)
    
    #Recomputing X2
    X2_new=forward_rnn(params, ut_train2, X2[-1],False,C)
    
    
    #Storing the X2
    X_concat=np.concatenate((X1_new,X2_new))
    X_concats.append(X_concat)
    
    #getting the Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
    X = X_concats[i]
    X_effective = X[washout:]
    yt_train=yt_targets[i]
    yt_train_effective = yt_train[washout:]
    # visualize_PCA_3D(X_effective,f' X training. T0=20,T1={T2[i]}') # If we want to visualize de X2 PCA
    params_trained, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
   
    ############################################################################
    
    #Checking if the CCL works in autonomous mode for the interpolation
    
    ########################################################################
    
    #Initial conditions for the scan X(0) and C(0)
    C_med = (C1 + C) / 2
    C_zeros=np.zeros((N,N))
    C_init=C_med
    X_init=None
    
    #interpolation conceptor for each T1, T2 combination
    interp=0 #interpolation coefficient, vary this for different validations
    C_interp=interp*C1+(1-interp)*Cs[i]
    C_interps.append(C_interp)
    
    #Computing X using C_interp as target
    X_interp,_=forward_rnn_CL(nu,a,beta,params_trained, ut_train2,C_init,C_interp,None,None,X_init,True)
    
    #computing the results and showing them
    mse_new=trained_model_new(X_interp,t,yt_train2,params_trained,step,washout,C_interp)   #Set yt, if you want to campare the results with something
    
    
    ##########################################################################
    
    #showing the results for more than one lambda
    
    #########################################################################
    
    visualize_sine_interpolation(t, ut_train2, params_trained, step, washout, C1, Cs[i],C_init,nu,a,beta)
    X_lambda,lam=visualize_sine_interpolation_one(t, ut_train2, params_trained, step, washout, C1, Cs[i],C_init,nu,a,beta)
    
    
    # ###########################################################################
    
    # #showing Paper's Fig 3, lambda scan
    
    # ###########################################################################
    
    
    X_scan,_=forward_rnn_CL(nu,a,beta,params_trained, t_interp,C_init,None,C1,Cs[i],X_init,True)
    
    #Showing X_scan
    visualize_PCA_3D(X_scan[washout:],f"X_scan: interpolation λ scan (0,1). T0=20,T1={T2[i]} ")
    
    # Output prediction
    Y_scan = X_scan @ params_trained['wout'].T + params_trained['bias_out'] 
    Y_scan = Y_scan[washout:]
    
    # Secondary axis values (λ)
    lam = np.linspace(0, 1, len(Y_scan))
    k = np.arange(len(Y_scan))

    fig, ax1 = plt.subplots()
    
    # First plot (x = k)
    ax1.plot(k, Y_scan)
    ax1.set_xlabel("k")
    ax1.set_ylabel("y(k)")
    ax1.set_title("Output interpolation λ scan")
    ax1.set_ylim([-1.5, 1.5])
    
    # Force scientific notation on bottom axis
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    
    # Second plot (x = λ) on a twin axis
    ax2 = ax1.twiny()
    ax2.plot(lam, Y_scan, color="none")  # don't show the second line
    # ax2.set_xlabel("λ")
    
    # Activate grid for both axes
    ax1.grid(True)
    # ax2.grid(True)
    
    plt.show()

    
    ##########################################################################
    
    #showing Paper's Fig 3, Periode lambda scan
    
    ###########################################################################
     
    
    # Parameters and input signal
    dt = 0.1  # sampling period
    t_scan = np.arange(len(Y_scan)) * dt
    Y_scan = Y_scan.flatten()
    
    #Showing the results
    T_instant(dt,t_scan,Y_scan) #here we use 3 different methods for the period calculation

