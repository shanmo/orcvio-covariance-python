import numpy as np 
import math
from matplotlib import pyplot as plt

MEDIUM_SIZE = 10
BIGGER_SIZE = 20
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

MSCKF_RESULT_FILE = "./cache/msckf/"
ORCVIO_RESULT_FILE = "./cache/orcvio/"

def get_cov_3c(covariances):
    """Obtain x, y, z axis 3 sigma bounds from covariance 
    """
    x_3c =  [3*math.sqrt(x[0, 0]) for x in covariances]
    y_3c =  [3*math.sqrt(x[1, 1]) for x in covariances]
    z_3c =  [3*math.sqrt(x[2, 2]) for x in covariances]
    return x_3c, y_3c, z_3c


if __name__ == "__main__": 

    file_name = 'msckf_covariances.npz'
    with np.load(MSCKF_RESULT_FILE + file_name) as data:
        covariances = data['arr_0']
    x_3c, y_3c, z_3c = get_cov_3c(covariances)
    
    n = len(covariances)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.plot(np.arange(n), x_3c, label='MSCKF', linestyle='dashed', linewidth=1.5, color='blue')
    ax1.plot(np.arange(n), [-x for x in x_3c], linestyle='dashed', linewidth=1.5, color='blue')
    ax1.set_title('x-axis 3c bounds')
    ax1.legend(loc="upper right")

    ax2.plot(np.arange(n), y_3c, label='MSCKF', linestyle='dashed', linewidth=1.5, color='blue')
    ax2.plot(np.arange(n), [-x for x in y_3c], linestyle='dashed', linewidth=1.5, color='blue')
    ax2.set_title('y-axis 3c bounds')
    ax2.legend(loc="upper right")

    ax3.plot(np.arange(n), z_3c, label='MSCKF', linestyle='dashed', linewidth=1.5, color='blue')
    ax3.plot(np.arange(n), [-x for x in z_3c], linestyle='dashed', linewidth=1.5, color='blue')
    ax3.set_title('z-axis 3c bounds')
    ax3.legend(loc="upper right")

    for ax in (ax1, ax2, ax3):
        ax.set(xlabel='Frame', ylabel='Meters')

    plt.show()