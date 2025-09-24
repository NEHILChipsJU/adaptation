#Imports
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from scipy.signal import find_peaks


from utils.rnn_utils import rnn_params
from utils.rnn_utils import forward_rnn
from utils.rnn_utils import forward_rnn_deg
from utils.rnn_utils import forward_rnn_mix_deg
from utils.rnn_utils import forward_rnn_CL_deg
from utils.utils import visualize_PCA_3D
from utils.utils import PCA_3D
from utils.rnn_utils import ridge
from utils.rnn_utils import compute_conceptor
from utils.utils import align_by_first_peak
from utils.utils import NRMSE
from utils.rnn_utils import trained_model_new

#Parameters that we want to acces trough the terminal

parser = argparse.ArgumentParser()
parser.add_argument("--m", type=int, default=10)
parser.add_argument("--random", type=int, default=70)
parser.add_argument("--threshold", type=float, default=0.1)
parser.add_argument("--reg", type=float, default=1)
parser.add_argument("--N", type=int, default=10)
parser.add_argument("--spectral_radius", type=float, default=1.6)
parser.add_argument("--scaling", type=float, default=0.9)
parser.add_argument("--bias_scaling", type=float, default=0.4)
parser.add_argument("--alpha", type=float, default=0.75)
parser.add_argument("--beta", type=float, default=0.4)
parser.add_argument("--nu", type=float, default=2.5e-5 )
parser.add_argument("--a", type=int, default=25 )
parser.add_argument("--seed", type=int, default=3) #seed for the random choice of degraded neurons
parser.add_argument("--T1", type=int, default=20)
parser.add_argument("--seedESN", type=int, default=1235) #seed for the random structure of the ESN
args = parser.parse_args()


#########################################################################

##Random inicialization of the ESN, with the parameters of the paper

########################################################################

spectral_radius=args.spectral_radius
scaling=args.scaling #Input scaling
bias_scaling=args.bias_scaling #bias
alpha=args.alpha #Leakage
a=args.a #Aperture.  
N=args.N # Network size 256
nu=args.nu #Learning Rate
beta=args.beta #Control gain
washout=20 # steps we wait until the network is stable, in order to show the results
reg=args.reg #regularization parameter in the Ride regression
step=1 # number of steps that the model will predict
sparsity=None
seed1=args.seed #random initialization
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
T1=args.T1 #fixed period
dt=T1/points
t=np.arange(0, 300,dt)

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
    seed=args.seedESN
)

#obtain matrix X1 (time, N) of internal states for all time points
X1=forward_rnn(params, ut_train1, x_init=None,autonomous=False,conceptor=None)
#Compute model conceptors
C1=compute_conceptor(X1, a)
#getting the final Wout with the ridge regression (Wout=Ytarget*X.T*(X*X.T+beta*I)^-1)
X_effective = X1[washout:]
yt_train_effective = yt_train1[washout:]
#showing training X
#training Wout with Xi 
params_trained, mse = ridge(reg, X_effective, yt_train_effective,step,params) #this gives us the results for the trainning dataset

####################################################################################

#Stabilization against partial network degradation

######################################################################################

#Setting the initial condicions
# C_init=(C1+C2)/2
C_init=np.eye(N,N)
X_init=None
C_t=C1

#keeping just the degraded part
last=int(len(ut_train1)/2)

#No degradation
PCA_nodeg=visualize_PCA_3D(X1[-last:],f"X for NO deg ")
trained_model_new(X1[washout:],ut_train1,yt_train1,params_trained,washout,True,None)

#parameters for the degradation
m=args.m #neurons degraded per step
random=args.random #number of trials
k=np.arange(0,int(0.3*args.N),m) #number of degraded nodes
np.random.seed(seed1) #if we want this seed to be alwas de same  
seed=np.random.randint(0, 2000, size=random) #seed for the random degradation

#storing the results
x_C=[] # storing 0 o 1, for succes or fail
x_CCL=[] # storing 0 o 1, for succes or fail
rate_C=[] # storing failure rate
rate_CCL=[] # storing failure rate
std_C=[] #standard desviation
std_CCL=[]

#Loop removing k nodes in each step
for i in tqdm(range(len(k))):
    for j in range(len(seed)):
        deg= (k[i]/N)*100 # % of degradation
           
        #C=cte with degradation
        X_cte_deg=forward_rnn_deg(params_trained, ut_train1,X_init, True,C1,deg,seed[j])
        PCA_C,_,_=PCA_3D(X_cte_deg[-last:],f"X for {deg}% deg with C=C1) ")
        
        #CCL  with degradation
        X_CL_deg,_=forward_rnn_CL_deg(nu,a,beta,params_trained, ut_train1,C_init,C_t,None,None,X_init,True,deg,seed[j]) #x% of degradation
        PCA_CCL,_,_=PCA_3D(X_CL_deg[-last:],f"X for {deg}% deg with CCL (separed) ")
        
        
        
        #failure or not
        if np.var(PCA_C)>args.threshold*np.var(PCA_nodeg): #setting the thershold for the failure
            x_C.append(0) #because we want to compute de failure rate
        else:
            x_C.append(1)
            
        if np.var(PCA_CCL)>args.threshold*np.var(PCA_nodeg):
            x_CCL.append(0)
        else:
            x_CCL.append(1)  
            
       
       
    # prob of x=1
    p_C = np.mean(x_C)
    p_CCL = np.mean(x_CCL)
    
    
    # standard desviatimn (Bernoulli)
    std_C.append( np.sqrt(p_C * (1 - p_C)))
    std_CCL.append(np.sqrt(p_CCL * (1 - p_CCL)))
   
    
    #failure rate
    if sum(x_C)==0:
        rate_C.append(0)        
    else:
       rate_C.append(sum(x_C)/len(x_C))  
    if sum(x_CCL)==0:
        rate_CCL.append(0) 
    else:
        rate_CCL.append(sum(x_CCL)/len(x_CCL))
    
        
    #clearing x because we want to compute the faliure rate for each step   
    x_C.clear()
    x_CCL.clear()
    
    
    
##################################################################################################################

# Qualitative evaluation

#################################################################################################################

#plotting the results
plt.figure()
plt.axhline(y=0.2, color="green", linestyle="--", label="Failure Rate 20%")
plt.axvline(x=0.1*args.N, color="black", linestyle="--", label="Degradation 10%")
plt.errorbar(k, rate_CCL, yerr=np.array(std_CCL)/2, fmt='o', color="blue", 
             ecolor="black", elinewidth=1, capsize=5, label="CCL")

plt.errorbar(k, rate_C, yerr=np.array(std_C)/2, fmt='o', color="red", 
             ecolor="black", elinewidth=1, capsize=5, label="C")

plt.xlabel("K")
plt.ylabel("Failure Rate")
plt.title("Qualitative evaluation")
plt.grid()
plt.legend()
# plt.savefig(f"plots/qualitative_rep{len(seed)}_th{args.threshold}_N{args.N}_reg{args.reg}_seed{args.seed}_a{args.a}_beta{args.beta}_nu{args.nu}_T{T1}_sESN{args.seedESN}_sr{args.spectral_radius}_alpha{args.alpha}_scaling{args.scaling}_bs{args.bias_scaling}.png", dpi=300, bbox_inches='tight')
plt.show()








