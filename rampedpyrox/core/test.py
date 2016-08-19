import numpy as np
from scipy import optimize

class Parameter:
    def __init__(self, value):
            self.value = value

    def set(self, value):
            self.value = value

    def __call__(self):
            return self.value

def fit(function, parameters, y, x = None):
    def f(params):
        i = 0
        for p in parameters:
            p.set(params[i])
            i += 1
        return y - function(x)

    if x is None: x = np.arange(y.shape[0])
    p = [param() for param in parameters]
    return optimize.leastsq(f, p)

def _phi_hat(eps):
    '''
    Calculates phi hat for given parameters.
    '''

    #generate Gaussian peaks
    y = _gaussian(eps,mus(),sigmas())

    #scale peaks to inputted height
    H = np.max(y,axis=0)
    y_scaled = y*heights()/H

    #calculate phi_hat
    phi_hat = np.sum(y_scaled,axis=1)

    return phi_hat

def _gaussian(x, mu, sigma):
    '''
    Calculates a Gaussian peak for a given x vector, mu, and sigma.
    '''

    #check data types and broadcast if necessary
    if isinstance(mu,(int,float)) and isinstance(sigma,(int,float)):
        #ensure mu and sigma are floats
        mu = float(mu)
        sigma = float(sigma)

    elif isinstance(mu,np.ndarray) and isinstance(sigma,np.ndarray):
        if len(mu) is not len(sigma):
            raise ValueError('mu and sigma arrays must have same length')

        #ensure mu and sigma dtypes are float
        mu = mu.astype(float)
        sigma = sigma.astype(float)

        #broadcast x into matrix
        n = len(mu)
        x = np.outer(x,np.ones(n))

    else:
        raise ValueError('mu and sigma must be float, int, or np.ndarray')

    #calculate scalar to make sum equal to unity
    scalar = (1/np.sqrt(2.*np.pi*sigma**2))

    #calculate Gaussian
    y = scalar*np.exp(-(x-mu)**2/(2.*sigma**2))

    return y

