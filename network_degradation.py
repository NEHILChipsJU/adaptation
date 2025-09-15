#Imports
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from utils.utils import T_instant
from utils.rnn_utils import rnn_params
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import forward_rnn_deg
from utils.rnn_utils import forward_rnn_mix_deg
from utils.rnn_utils import forward_rnn_mix
from utils.rnn_utils import forward_rnn_CL_deg
from utils.rnn_utils import forward_rnn_interp
from utils.rnn_utils import forward_rnn_CL
# from utils.rnn_utils import forward_rnn_CL_mix
from utils.utils import visualize_PCA_3D
from utils.utils import visualize_multiple_PCA_3D
from utils.rnn_utils import ridge
from utils.rnn_utils import compute_conceptor
from utils.utils import visualize_sine_interpolation
from utils.utils import visualize_sine_interpolation_one
# from utils.rnn_utils import degradation
from utils.rnn_utils import trained_model_new


#########################################################################

##Random inicialization of the ESN, with the parameters of the paper

########################################################################

spectral_radius=1.6
scaling=1
bias_scaling=1
alpha=0.75 #Leakage
a=25 #Aperture.  
N=256 # Network size 256
nu=2.5e-5 #Learning Rate
beta=0.2 #Control gain
washout=20 # steps we wait until the network is stable, in order to show the results
reg=1 #regularization parameter in the Ride regression
step=1 # number of steps that the model will predict
sparsity=None
deg= (9/N)*100 # % of degradation
# deg=90


#Input Signal
def sin(T,t, A):
    """Sinus wave with a given period T, and a time series"""
    s=A*np.sin((2*np.pi*t)/T)
    return s

#Temporal Series
points=40 #points in one periode
A=1 #Sinus Amplitude
# now we will use more than one input 
T1=20 #fixed period
dt=T1/points
t=np.arange(0, 300,dt)
# plt.plot(sin(T1,t,A),'.-')
# plt.xlim([0,45])
#obtaining the input and some parameters for params
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
#getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
X_effective = X1[washout:]
yt_train_effective = yt_train1[washout:]
#showing training X
#training Wout with Xi applying Ci
params_trained, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset

####################################################################################

#Stabilization against partial network degradation

######################################################################################

#Setting a better C_init
#define a variable with the training data and the targets (to predict the last point of the time series)
T2=30
data2=sin(T2,t,A)
data2=data2.reshape(-1, 1)
#create shifted input and output, the output is the target
ut_train2 = data2[:-step]               # shape (N-step, 1)
yt_train2 = data2[step:]                # shape (N-step, 1)
##Extracting X for each T2
X2=forward_rnn(params, ut_train2, x_init=None,autonomous=False,conceptor=None)
C2=compute_conceptor(X2, a)



# C_init=(C1+C2)/2
C_init=np.eye(N,N)
X_init=None
C_t=C1


last=int(len(ut_train1)/3)
#computing the results and showing them
#No degradation

PCA_nodeg=visualize_PCA_3D(X1[-last:],f"X for NO deg ")
mse_cte=trained_model_new(X1[washout:],ut_train1,yt_train1,params_trained,washout,True,None)

#CCL (separed) no degradation
X_CL,_=forward_rnn_CL(nu,a,beta,params_trained, ut_train1,C_init,C_t,None,None,X_init,True) #no degradation
visualize_PCA_3D(X_CL,f"X for NO deg with CCL (separed) ")
mse_CL=trained_model_new(X_CL[washout:],ut_train1,yt_train1,params_trained,washout,True,None)

# CCL (mixed) no degradation
X_CL_mix,_=forward_rnn_mix(nu,a,beta,params_trained, ut_train1,C_init,C_t,None,None,X_init,True) #no degradation
visualize_PCA_3D(X_CL_mix,f"X for NO deg with CCL (mixed) ")
mse_CL_mix=trained_model_new(X_CL_mix[washout:],ut_train1,yt_train1,params_trained,True,washout,None)
    


#C=cte with degradation
X_cte_deg=forward_rnn_deg(params_trained, ut_train1,X_init, True,C1,deg)
PCA_C=visualize_PCA_3D(X_cte_deg[-last:],f"X for {deg}% deg with C=C1) ")
mse_cte_deg=trained_model_new(X_cte_deg[washout:],t,yt_train1,params_trained,washout, True,C1)

#CCL (separed) with degradation
X_CL_deg,_=forward_rnn_CL_deg(nu,a,beta,params_trained, ut_train1,C_init,C_t,None,None,X_init,True,deg) #x% of degradation
PCA_CCL=visualize_PCA_3D(X_CL_deg[-last:],f"X for {deg}% deg with CCL (separed) ")
mse_cl_deg=trained_model_new(X_CL_deg[washout:],t,yt_train1,params_trained,washout,True,None)


#CCL (mixed) with degradation
X_CL_deg_mix,_=forward_rnn_mix_deg(nu,a,beta,params_trained, ut_train1,C_init,C_t,None,None,X_init,True,deg) #x% of degradation
visualize_PCA_3D(X_CL_deg_mix[-last:],f"X for {deg}% deg with CCL (mixed) ")
mse_cl_dmix=trained_model_new(X_CL_deg_mix[washout:],t,yt_train1,params_trained,washout,True,None)




X=[X1,X_CL_deg[-last:],X_cte_deg[-last:]]
label=['No Deg','CCL','C']
visualize_multiple_PCA_3D(X,label)

Xt=[X1,X_CL_deg[-last:],X_cte_deg[-last:],X_CL_deg_mix[-last:]]
labelt=['No Deg','CCL(separed)','C','CCL(mixed)']
visualize_multiple_PCA_3D(Xt,labelt)

