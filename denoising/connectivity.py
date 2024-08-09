import numpy as np
from scipy import stats
from gglasso.problem import glasso_problem
from nilearn.connectome import ConnectivityMeasure

                
def functional_connectivity(data, measure='correlation', **kwargs):
    """
    ['glasso', "covariance", "correlation", 
                        "partial correlation", "tangent", "precision"]
    """
    
    if measure == 'glasso':
        return glasso(data, **kwargs)
    else:
        return con_measure(data, measure)


def con_measure(ts, measure):
    """
    Functional connectivity calculation

    Parameters
    ----------
    ts: list, np.array
        List of np.arrays or np.array of shape (2, :, :) or (3, :, :)
    measure: str
        One of ["covariance", "correlation", "partial correlation", "tangent", "precision"]. By default "correlation"

    Returns
    -------
    list of np.arrays or np.array
    """

    connectivity_measure = ConnectivityMeasure(kind=measure, standardize=False)
    fc = []
    if isinstance(ts[0], list):
        for l in ts:
            calc = connectivity_measure.fit_transform(l)
            for i in calc:
                np.fill_diagonal(i, 0)
            fc.append(calc)
        fc = np.array(fc)

    elif (isinstance(ts, np.ndarray) and len(ts.shape) == 2):
        fc = connectivity_measure.fit_transform([ts])
        for i in fc:
            np.fill_diagonal(i, 0)

        if fc.shape[0] == 1:
            fc = np.squeeze(fc)

    elif isinstance(ts, np.ndarray) and len(ts.shape) == 3:
        fc = connectivity_measure.fit_transform(ts)
        for i in fc:
            np.fill_diagonal(i, 0)

        if fc.shape[0] == 1:
            fc = np.squeeze(fc)

    elif isinstance(ts, list):
        fc = connectivity_measure.fit_transform(ts)
        for i in fc:
            np.fill_diagonal(i, 0)

    return fc   


def glasso(data, is_scaled=False, L1=0.001):
    """
    Calculates the L1-regularized partial correlation matrix of a dataset. 

    INPUT:
        data : a dataset with dimension [nNodes x nDatapoints]
        L1 : L1 (lambda1) hyperparameter value
    OUTPUT:
        glassoParCorr : regularized partial correlation coefficients (i.e., FC matrix)
        prec : precision matrix, where entries are not yet transformed into partial correlations (used to compute loglikelihood)
    """

    data = np.transpose(data, (0, 2, 1))
    if not is_scaled:
        data_scaled = stats.zscore(data, axis=1)

    outp = np.zeros((
        data.shape[0],
        data.shape[1],
        data.shape[1]))
    
    for i in range(len(data_scaled)):
        empCov = np.cov(data_scaled[i], rowvar=True)

        glasso = glasso_problem(empCov, data.shape[1], 
                                reg_params={'lambda1': L1}, 
                                latent=False, do_scaling=False)
        glasso.solve(verbose=False)
        prec = np.squeeze(glasso.solution.precision_)

        # Transform precision matrix into regularized partial correlation matrix
        denom = np.atleast_2d(1. / np.sqrt(np.diag(prec)))
        glassoParCorr = -prec * denom * denom.T
        np.fill_diagonal(glassoParCorr, 0)

        outp[i] = glassoParCorr

    return outp

def glasso_ms(data, is_scaled=False, L1=0.001):
    """
    Calculates the L1-regularized partial correlation matrix of a dataset. 

    INPUT:
        data : a dataset with dimension [nNodes x nDatapoints]
        L1 : L1 (lambda1) hyperparameter value
    OUTPUT:
        glassoParCorr : regularized partial correlation coefficients (i.e., FC matrix)
        prec : precision matrix, where entries are not yet transformed into partial correlations (used to compute loglikelihood)
    """

    data = np.transpose(data, (0, 2, 1))
    if not is_scaled:
        data_scaled = stats.zscore(data, axis=1)

    outp = []
    
    for i in range(5, 15):
        empCov = np.cov(data_scaled[i], rowvar=True)

        glasso = glasso_problem(empCov, data.shape[1], 
                                reg_params={'lambda1': L1}, 
                                latent=False, do_scaling=False)
        
        L1s = np.arange(-.5, -3.1, -.2)
        L1s = np.round(10 **L1s, 6)
        modelselect_params = {'lambda1_range': L1s}
        glasso.model_selection(modelselect_params=modelselect_params)
        
        outp.append(glasso.reg_params['lambda1'])


    return outp