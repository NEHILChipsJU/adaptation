import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks, stft, savgol_filter




from utils.utils import T_instant
from utils.rnn_utils import rnn_params
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import forward_rnn_interp
from utils.utils import visualize_PCA_3D
from utils.rnn_utils import ridge
from utils.rnn_utils import trained_model_new
from utils.rnn_utils import compute_conceptor
from utils.utils import visualize_sine_interpolation
from utils.utils import visualize_sine_interpolation_one
##Random inicialization of the ESN, with the parameters of the paper

spectral_radius=1.6
scaling=1
bias_scaling=1
alpha=0.75 #Leakage
a=25 #Aperture.  
N=256 # Network size
 # nu=0.2 #Learning Rate
# beta=2.5e-5 #Control gain
washout=20 # steps we wait until the network is stable, in order to show the results
reg=0.0001 #regularization parameter in the Ride regression
step=1 # number of steps that the model will predict
sparsity=None
#time steps for the Fig 3 interpolation
n_steps=150000
t_interp=np.arange(0,n_steps-1,1)

#Input Signal
def sin(T,t, A):
    """Sinus wave with a given period T, and a time series"""
    s=A*np.sin((2*np.pi*t)/T)
    return s
#Temporal Series
# dt=0.1
# t=np.arange(0, 300,dt)
A=1 #Sinus Amplitude
# now we will use more than one input 
T1=20 #fixed period
T2=[25,30,35] #variable period, we want to compare differents T2 with the same T1
points=40 #number of points in each period
dt=T1/points
t=np.arange(0, 300,dt)
plt.plot(sin(T1,t,A),'.-')
plt.xlim([0,250])
#########For Fixed T#############3
data1=sin(T1,t,A)
data1=data1.reshape(-1, 1)
#create shifted input and output, the output is the target
ut_train1 = data1[:-step]               # shape (N-step, 1)
yt_train1 = data1[step:]                # shape (N-step, 1)
#get dimensions
input_size=ut_train1.shape[-1]
output_size=yt_train1.shape[-1]
#build a dictionary with the parameters that will form the ESN, so we have them organized
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
#obtain matrix X1 (time, N) of internal states for all time points
X1=forward_rnn(params, ut_train1, x_init=None,autonomous=False,conceptor=None)
#Compute model conceptors
C1=compute_conceptor(X1, a)
#Re computing X2
X1_new=forward_rnn(params, ut_train1, None,True,C1)
####### For Variable T #########
#generating the palce where the differents concat will be stored
Y=[]
X_concats = []
yt_targets = []
Cs=[]
C_interps=[]
ut_concats=[]
y_lams=[]
Y_scans=[]
for i in range(len(T2)):
    #Showing the sinus
    plt.figure()
    plt.plot(t,sin(T1,t,A),label=f"T1 = {T1}")
    plt.plot(t,sin(T2[i],t,A),label=f"T2 = {T2[i]}")
    plt.legend()
    
    #define a variable with the training data and the targets (to predict the last point of the time series)
    data2=sin(T2[i],t,A)
    data2=data2.reshape(-1, 1)
    #create shifted input and output, the output is the target
    ut_train2 = data2[:-step]               # shape (N-step, 1)
    yt_train2 = data2[step:]                # shape (N-step, 1)
    ##Extracting X for each T2
    X2=forward_rnn(params, ut_train2, x_init=None,autonomous=False,conceptor=None)
    
    ut_concat=np.concatenate((ut_train1,ut_train2))
    #X_concat=np.concatenate((X1,X2))
    yt_target=np.concatenate((yt_train1,yt_train2))
    ut_concats.append(ut_concat)
    #X_concats.append(X_concat)
    yt_targets.append(yt_target)
    #Compute T2 conceptors
    C=compute_conceptor(X2, a)
    Cs.append(C)
    #Recomputing X2
    X2_new=forward_rnn(params, ut_train2, X2[-1],False,C)
    X_concat=np.concatenate((X1,X2))
    X_concats.append(X_concat)
    #getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
    X = X_concats[i]
    X_effective = X[washout:]
    yt_train=yt_targets[i]
    yt_train_effective = yt_train[washout:]
    #training Wout with Xi applying Ci
    params_trained, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset
   
    X1_new=forward_rnn(params_trained, ut_train1, None,True,C1)
    if i==0:
        Y1 = X1_new @ params_trained['wout'].T + params_trained['bias_out'] 
        Y.append(Y1)
    
    ###############Showing if T0 or T1 can be generated with the C_interp################################
    X_init=None
    #interpolation conceptor for each T1, T2 combination
    interp=0 #interpolation coefficient
    C_interp=interp*C1+(1-interp)*Cs[i]
    C_interps.append(C_interp)
    
    # computing the target
    T3=T1*interp +(1-interp)*T2[i]
    #define a variable with the training data and the targets (to predict the last point of the time series)
    data3=sin(T3,t,A)
    data3=data3.reshape(-1, 1)
    #create shifted input and output, the output is the target
    ut_train3 = data3[:-step]               # shape (N-step, 1)
    yt_train3 = data3[step:]                # shape (N-step, 1)
    #computing the results and showing them
    X_cte=forward_rnn(params_trained, ut_train3,X_init, True,C_interp)
    mse_cte=trained_model_new(X_cte[washout:],ut_train3,yt_train3,params_trained,washout,True,C_interp)   #Set yt, if you want to campare the results with something
    
    Y2 = X_cte @ params_trained['wout'].T + params_trained['bias_out'] 
    Y.append(Y2)
    
    ###################showing the results for more than one lambda#########################
    # visualize_sine_interpolation(t, ut_train2, yt_train2, params_trained, step, T2, A, washout, C1, Cs[i])
    # visualize_sine_interpolation_one(t, ut_train2, yt_train2, params_trained, step, T2, A, washout, C1, Cs[i])

    ####################showing Paper's Fig 3, lambda scan####################################
    
    X_scan=forward_rnn_interp(params_trained, t_interp,C1,Cs[i],X_init)
    # Output prediction
    Y_scan = X_scan @ params_trained['wout'].T + params_trained['bias_out'] 
    Y_scan = Y_scan[washout:]
    Y_scans.append(Y_scan)
    
    
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

    
    
    
    
   ############### Instantaneous phase  ###################
  
    # ==============================
    # Parameters and input signal
    # ==============================
    dt = T1/points  # sampling period
    t_scan = np.arange(len(Y_scan)) * dt
    Y_scan = Y_scan.flatten()
    
    #Shoeing the results
    T_instant(dt,t_scan,Y_scan)
    
    
    

for i in range(len(T2)):
    plt.figure()
    # Secondary axis values (λ)
    lam = np.linspace(0, 1, len(Y_scan))
    k = np.arange(len(Y_scan))

    fig, ax1 = plt.subplots()
    
    # First plot (x = k)
    ax1.plot(k, Y_scans[i],label=fr"$T_2={T2[i]}$")
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
    ax1.legend(frameon=False)
    ax1.grid(True)
    # ax2.grid(True)
    
    plt.show()


#Showing interpolation together
fig, ax1 = plt.subplots()

for i in range(len(T2)):
    # Secondary axis values (λ)
    lam = np.linspace(0, 1, len(Y_scans[i]))
    k = np.arange(len(Y_scans[i]))

    # First plot (x = k) with transparency
    ax1.plot(k, Y_scans[i], label=fr"$T_2={T2[i]}$")

# Axis settings
ax1.set_xlabel("k")
ax1.set_ylabel("y(k)")
ax1.set_title(fr"Output interpolation λ scan $T_0=20$")
ax1.set_ylim([-1.5, 1.5])

# Force scientific notation on bottom axis
ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

# Second axis (λ) on top
ax2 = ax1.twiny()
ax2.set_xlim(0, 1)  # since λ goes from 0 to 1
ax2.set_xlabel("λ")

# Legend and grid
ax1.legend(frameon=False, loc="upper center",  ncol=len(T2))
ax1.margins(x=0)
ax1.grid(True)

plt.show()

#showing T together
plt.figure()
for i in range(len(T2)):    
    ##################################################################
     
    #  Peak-to-Peak Method
     
    ####################################################################
    k = np.arange(len(Y_scans[i]))
    peaks, _ = find_peaks(np.ravel(-Y_scans[i]), height=0)  # Detect index peaks
    t_peaks = t_scan[peaks] #time peaks
     
    # Instantaneous period = difference between consecutive peaks
    period_p2p = np.diff(t_peaks) 
    t_p2p = t_peaks[:-1] + np.diff(t_peaks)/2
     
    # Smooth trend
    period_p2p_smooth = savgol_filter(period_p2p, 51, 3)
    
    #plotting
    t_draw=np.linspace(0,n_steps-1,len(period_p2p_smooth))
    plt.plot(t_draw,period_p2p_smooth,label=fr"$T_2={T2[i]}$")
plt.xlabel("k")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# plt.ylim([18,40])
plt.ylabel("Instantaneous Period T(k)")
plt.margins(x=0)
plt.grid()
plt.legend()
plt.show()
   
 
#showing the sinus generated with C
plt.figure()
for i in range(len(Y)):    
    if i==0:
        plt.plot(Y[i],'black',label=fr"$C_0$")
    else:
        plt.plot(Y[i],label=fr"$C_1$, $T_2={T2[i-1]}$")
    
plt.xlabel("k")
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
# plt.ylim([18,40])
plt.ylabel("y(k)")
plt.xlim([50,300])
plt.margins(x=0)
plt.grid()
plt.legend()
plt.show()
















# for interp in np.linspace(0, 1,100):
#     X_init=None
#     #interpolation conceptor for each T1, T2 combination
#     # interp=0.25 #interpolation coefficient
#     C_interp=interp*C1+(1-interp)*Cs[i]
#     C_interps.append(C_interp)
    
#     # computing the target
#     T3=T1*interp +(1-interp)*T2[i]
#     #define a variable with the training data and the targets (to predict the last point of the time series)
#     data3=sin(T3,t,A)
#     data3=data3.reshape(-1, 1)
#     #create shifted input and output, the output is the target
#     ut_train3 = data3[:-step]               # shape (N-step, 1)
#     yt_train3 = data3[step:]                # shape (N-step, 1)
#     #computing the results and showing them
#     X_cte=forward_rnn(params_trained, ut_train3,X_init, True,C_interp)
#     mse_cte=trained_model_new(X_cte[washout:],ut_train3,yt_train3,params_trained,washout,True,C_interp)   #Set yt, if you want to campare the results with something







