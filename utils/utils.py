import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks, stft, savgol_filter

from utils.rnn_utils import forward_rnn
from utils.rnn_utils import forward_rnn_CL
   


def visualize_sine_interpolation(
    t, ut, params_trained, step, washout, C1, C2, C_init=None ,nu=None, a=None,beta=None
):
    """
    Visualizes the model output interpolation for several lambda values
    in vertical subplots (one per lambda).
    
    Args:
    - t (array): time series
    - ut (array): input signal
    - N (int): rnn size    
    - params_trained (dict): dictionary containing the RNN parameters (weights and biases) trained.
    - step (int): Number of steps of the prediction
    - washout (int): number of steps for the washout
    - C1 (array): conceptor for T1
    - C2 (array): conceptor for T2
    - C_init (array): conceptor for t=0
    - nu (float): learning rate
    - a (int): aperture
    - beta (float): control gain

    Returns:
    - plot
    """
    lambdas = [0, 0.25, 0.5, 0.75, 1]

    # Adjust the data after washout
    t = t[washout + step:]
    
    # Create vertical subplots: one per lambda
    # fig, axs = plt.subplots(len(lambdas), 1, figsize=(10, 8), sharex=True)
    fig, axs = plt.subplots(len(lambdas), sharex=True, sharey=True)
    for idx, lamda in enumerate(lambdas):
        # Interpolate the conceptor
        C_interp = lamda * C1 + (1 - lamda) * C2

        # Compute hidden states
        if nu is None:
            X = forward_rnn(params_trained, ut, None, True, C_interp)
            title=', C'

        else:
            X,_=forward_rnn_CL(nu,a,beta,params_trained, ut,C_init,C_interp,None,None,None,True)
            title= ', CCL'
        X = X[washout:]

        # Compute prediction
        Y_pred = X @ params_trained['wout'].T + params_trained['bias_out']

        # Plot
        ax = axs[idx]
        ax.plot(t, Y_pred, linewidth=2)
        ax.set_ylabel("y(k)")
        ax.set_ylim([-1.5, 1.5])
        ax.grid(True, alpha=0.3)

        # Inline label instead of legend
        ax.text(
            0.98, 0.82, rf"$\lambda = {lamda}$",
            transform=ax.transAxes, ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8)
        )

        # Clean up axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axs[-1].set_xlabel("t(s)")

    # Title with enough space above
    fig.suptitle("Output interpolation for different λ" + title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()




def visualize_sine_interpolation_one(
    t, ut, params_trained, step, washout, C1, C2, C_init=None,nu=None, a=None,beta=None
):
    """
    Visualizes the output interpolation of the trained model 
    for several lambda values in a single figure with multiple curves.
    Args:
    - t (array): time series
    - ut (array): input signal
    - N (int): rnn size    
    - params_trained (dict): dictionary containing the RNN parameters (weights and biases) trained.
    - step (int): Number of steps of the prediction
    - washout (int): number of steps for the washout
    - C1 (array): conceptor for T1
    - C2 (array): conceptor for T2
    - C_init (array): conceptor for t=0
    - nu (float): learning rate
    - a (int): aperture
    - beta (float): control gain

    Returns:
    - plot
    """
    # List of lambda values
    lambdas = [0, 0.25, 0.5, 0.75, 1]

    # Adjust data after the washout period
    t = t[washout + step:]

    # Create a single figure
    plt.figure(figsize=(10, 6))
    #storing X
    X_stored=[]
    
    for lamda in lambdas:
        # Conceptor interpolation
        C_interp = lamda * C1 + (1 - lamda) * C2

         # Compute hidden states
        if nu is None: #if there is a nu, it means that we want to use de CCL
            X = forward_rnn(params_trained, ut, None, True, C_interp)
            title= 'Output interpolation for different λ values'
        else:
            X,_=forward_rnn_CL(nu,a,beta,params_trained, ut,C_init,C_interp,None,None,None,True)
            title= 'Output interpolation for different λ values, CCL'
        X = X[washout:]
        X_stored.append(X)
        # Output prediction
        Y_pred = X @ params_trained['wout'].T + params_trained['bias_out']

        # Plot the curve for this lambda
        plt.plot(t, Y_pred, label=fr"$\lambda={lamda}$")
        
    #Rest of the plot
    plt.xlabel("t(s)")
    plt.ylabel("y(t)")
    plt.title(title)
    plt.ylim([-1.5, 1.5])
    plt.legend(frameon=False)
    plt.grid(True)
    plt.show()
        
    return X_stored, lambdas
    
    




def T_instant(dt,t_scan,Y_scan):
    """
    Computing and visualizing the output period with 3 different methods
    Args:
    - dt (float): sampling scale
    - t_scan (array): time series
    - Y_scan (array): scan output
    

    Returns:
    - plot
    """
    ####################################################
    
    #  Instantaneous Period using Hilbert Transform
    
    ####################################################
    analytic_signal = hilbert(Y_scan)
    phase = np.unwrap(np.angle(analytic_signal))
    freq_inst = np.diff(phase) / (2.0 * np.pi) * (1/dt)
    period_inst = 1 / freq_inst
    
    # Remove edges to avoid artifacts
    t_scan_diff = t_scan[1:]
    margin = int(0.05 * len(period_inst))
    t_valid = t_scan_diff[margin:-margin]
    period_valid = period_inst[margin:-margin]
    
    # Smooth trend with Savitzky-Golay filter
    period_hilbert_smooth = savgol_filter(period_valid, 301, 3) #reducing the noise
    
   ####################################################################
    
    #  Peak-to-Peak Method
    
    ####################################################################
    peaks, _ = find_peaks(Y_scan, height=0)  # Detect index peaks
    t_peaks = t_scan[peaks] #time peaks
    
    # Instantaneous period = difference between consecutive peaks
    period_p2p = np.diff(t_peaks) 
    t_p2p = t_peaks[:-1] + np.diff(t_peaks)/2
    
    # Smooth trend
    period_p2p_smooth = savgol_filter(period_p2p, 51, 3)
    
    #####################################################################
    
    #  STFT (Short-Time Fourier Transform)
   
    ###################################################################
    # Improved STFT
    fs = 1/dt
    f, t_stft, Zxx = stft(Y_scan, fs=fs, nperseg=1024, noverlap=800)  # longer window
    magnitude = np.abs(Zxx)
    
    idx_max = np.argmax(magnitude, axis=0)
    freq_dominant = f[idx_max]
    period_stft = 1 / freq_dominant
    
    # Remove NaN or inf values, for central time
    period_stft = np.nan_to_num(period_stft, nan=np.mean(period_stft[1:]))
    
    # Remove the first 3 windows (edge effects)
    t_stft = t_stft[3:]
    period_stft = period_stft[3:]
    
    # Smooth with Savitzky-Golay
    period_stft_smooth = savgol_filter(period_stft, 11, 3)
    
    
    
    
   
    # Plot Comparison (2 Methods)
    plt.figure()
    plt.plot(t_valid, period_hilbert_smooth, 'r', label="Hilbert (smoothed)")
    plt.plot(t_p2p, period_p2p_smooth, 'g', label="Peak-to-Peak (smoothed)")
    plt.xlabel("k")
    plt.ylabel("Instantaneous Period T(k)")
    plt.title("Instantaneous Period Comparison (2 Methods)")
    plt.grid()
    plt.legend()
    plt.show() 

    
def visualize_PCA_3D(X,label):
    """
    Visualize the PCA of any matrix, and project other the conceptor in the PCA space
    Args:
    - X: (numpy.ndarray): Input matrix 
    - lable: text for the plot title
    
    Returns:
    - plot
    """
    # centering the X
    X_centered = X - np.mean(X, axis=0)
    
    #Obtainning the PCA components
    pca = PCA(n_components=3)  # number of axis
    X_pca = pca.fit_transform(X_centered)
    #storing the PCA1
    PCA1=X_pca[:,0]
    #Plotting the PCA
    fig = plt.figure(figsize=(12, 8))  
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.5, label='X projected')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"PCA 3D, {label}")
    plt.grid(True)
    plt.legend()
    plt.show() 
    
    return PCA1

     
        

def visualize_multiple_PCA_3D(X,label):
    """
    Visualize the PCA of any matrix, and project other the conceptor in the PCA space
    Args:
    - X: (list): Input matrix list
    - label (list): text for the plot title
    
    Returns:
    - plot
    """
    
    
    #Plotting the PCA
    fig = plt.figure(figsize=(12, 8))  
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(X)):
        # centering the X
        X_centered = X[i] - np.mean(X[i], axis=0)
        
        #Obtainning the PCA components
        pca = PCA(n_components=3)  # number of axis
        X_pca = pca.fit_transform(X_centered)
        ax.plot3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.5, label=fr'{label[i]}')
        
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    # ax.set_title(f"PCA 3D")
    plt.grid(True)
    plt.legend()
    plt.show()      
    
    
def PCA_3D(X,label):
    """
    Visualize the PCA of any matrix, and project other the conceptor in the PCA space
    Args:
    - X: (numpy.ndarray): Input matrix 
    - lable: text for the plot title
    
    Returns:
    - PCA components
    """
    # centering the X
    X_centered = X - np.mean(X, axis=0)
    
    #Obtainning the PCA components
    pca = PCA(n_components=3)  # number of axis
    X_pca = pca.fit_transform(X_centered)
    #storing the PCA1
    PCA1=X_pca[:,0]
    PCA2=X_pca[:,1]
    PCA3=X_pca[:,2]
    
    
    
    return PCA1, PCA2, PCA3


def align_by_first_peak(y1, y2, prominence=None, height=None, distance=None):
    
    """
    phase alignment by first peak shift
    - y1 (numpy.ndarray): Reference output 
    - y2 (numpy.ndarray): Output that we want to align with y1
    
    Returns:
    - y2_alined (array): y2 aligned with y1
    - shift (ind): difference between y1 and y2 indexs
    """
    
    #transforming y1 and y2 
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    y1=np.ravel(y1)
    y2=np.ravel(y2)
    #computing all the peaks
    peaks1, _ = find_peaks(y1, prominence=prominence, height=height, distance=distance)
    peaks2, _ = find_peaks(y2, prominence=prominence, height=height, distance=distance)
    
    #peaks index

    i1 = peaks1[0]
    i2 = peaks2[0]

    # aligning y2  with y1
    shift = i2 - i1         
    y2_aligned = np.roll(y2, -shift)

    return y2_aligned,shift


def NRMSE(y1,y2):
    """
    Normalized Root Mean Square Error
    
    phase alignment by first peak shift
    - y1 (numpy.ndarray): Reference output 
    - y2 (numpy.ndarray):Predicted output
    
    Returns:
    - nrsme: Normalized Root Mean Square Error between y1 and y2

    """
    y1=np.ravel(y1)
    y2=np.ravel(y2)
    
    #fist computing mse
    mse=np.mean((y2 - y1)**2)
    
    #secondly rmse
    rmse=np.sqrt(mse)
    
    #computing the mean of y1
    mean=np.mean(y1)
    std=np.std(y1)
    
    rang=np.max(y1)-np.min(y1)
    
    # finally the nrmse by mean
    nrmse=rmse/std
    
    return nrmse

