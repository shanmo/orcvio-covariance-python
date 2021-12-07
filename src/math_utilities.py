import numpy as np

def skew(vec):
    """
    Create a skew-symmetric matrix from a 3-element vector.
    """
    vec = np.squeeze(vec)
    if vec.size != 3:
        raise TypeError("Input size to skew operator must be 3")
    x, y, z = vec
    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]])

def symmeterize_matrix(matrix):
    """Makes sure that the matrix is symmetric along the diagonal.

    Args:
        matrix: NxN matrix to make symmetric. Must be square.

    Returns:
        NxN matrix that is symmetric along the diagonal.


    A Kalman Filter expects that the covariance matrix is symmetric. However, due to numerical problems this may not
    always be the case as it gets modifed as the algorithm is run. Thus you can call this function after a modification
    (e.g. an Update or propogation) and make sure that it is symmetric.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise TypeError("Matrix must be square(equal rows and columns)")
    return (matrix + matrix.T) / 2

def odotOperator(ph):
    '''
    @Input:
      ph = n x 4 = points in homogeneous coordinates
    @Output:
    odot(ph) = n x 4 x 6
    '''

    zz = np.zeros(ph.shape + (6,))
    zz[...,:3,3:6] = -skew(ph[...,:3])
    zz[...,0,0],zz[...,1,1],zz[...,2,2] = ph[...,3],ph[...,3],ph[...,3]

    return zz

def Hl_operator(omega):
    """
    implements Hl operator in eq 20 
    """

    omega_norm = np.linalg.norm(omega) 

    term1 = (1/2)*np.eye(3)

    if (omega_norm < 1.0e-5):
        return term1 

    term2 = np.nan_to_num((omega_norm - np.sin(omega_norm)) / (omega_norm**3)) * skew(omega)
    term3 = np.nan_to_num((2*(np.cos(omega_norm) - 1) + (omega_norm**2)) / (2*(omega_norm**4))) * (skew(omega) @ skew(omega))

    Hl = term1 + term2 + term3 
    
    return Hl

def Jl_operator(omega):
    """
    implements Jl operator in eq 20 
    """

    omega_norm = np.linalg.norm(omega) 

    term1 = np.eye(3)

    if (omega_norm < 1.0e-5):
        return term1 

    term2 = np.nan_to_num((1 - np.cos(omega_norm)) / (omega_norm**2)) * skew(omega)
    term3 = np.nan_to_num((omega_norm - np.sin(omega_norm)) / (omega_norm**3)) * (skew(omega) @ skew(omega))

    Jl = term1 + term2 + term3

    return Jl

def get_cam_wrt_imu_se3_jacobian(cRi, iPc, cRw):

    p_cxi_p_ixi = np.zeros((6, 6))

    p_cxi_p_ixi[0:3, 0:3] = -1 * cRi @ skew(iPc)
    p_cxi_p_ixi[3:6, 0:3] = cRi
    p_cxi_p_ixi[0:3, 3:6] = cRw

    return p_cxi_p_ixi