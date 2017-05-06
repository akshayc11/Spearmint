import sys
import traceback
import random
import numpy as np
import math

def recency_weights(num):
    # This function generates an exponential based recency weighting to add to the residuals.
    # This is done so that the loss from fresher epochs are given more weightage than loss from 
    # previous epochs
    if num == 1:
        return np.ones(1)
    else:
        recency_weights = [10**(1./num)] * num
        recency_weights = recency_weights ** (np.arange(0, num))
        return recency_weights
    pass

def gradient_descent(f_cs, f_ps,
                     alpha=0.01,
                     N=1000,
                     lambda_1=0,
                     lambda_2=100,
                     lambda_3=1,
                     lambda_4=1,
                     lambda_5=5,
                     lambda_hp=0.0,
                     scale_hp=1,
                     f_p_cov=None,
                     return_loss=False,
                     a_0=None,
                     b_0=None,
                     recency_weighting=False,
                     monotonicity_condition=False,
                     f_c_max=None,
                     f_p_max=None):
    if monotonicity_condition is True:
        assert f_c_max is not None and f_p_max is not None, "Need f_c_max and f_p_max to check monotonicity condition"
        lambda_6=1
    else:
        f_c_max = 0
        f_p_max = 0
        lambda_5 = 0
        lambda_6 = 0
    if a_0 is None:
        a = random.random()
    else:
        a = a_0
    if b_0 is None:
        b = random.random() - 0.5
    else:
        b = b_0

    if f_p_cov is None:
        f_p_cov = 1.0 / scale_hp

    n = len(f_cs)
    f_cap_cs = a * f_ps + b
    try:
        assert(len(f_cs) == len(f_ps))
        for i in xrange(N):
            f_cap_cs = a*f_ps + b
            residuals = f_cs - f_cap_cs
            if recency_weighting is True:
                residuals = np.sqrt(recency_weights(n)) * residuals
            
            monotonicity_val = lambda_5*((a * f_p_max) + b - f_c_max)
            
            dL_da = -np.dot(residuals, f_ps)/n - (lambda_1 * lambda_2 * math.exp(-lambda_2 * a)) - (lambda_3 * (1.0 - a) / math.exp(lambda_4 * n)) - (lambda_6 * lambda_5 * f_p_max * math.exp(-monotonicity_val))
            dL_db = -np.sum(residuals)/n - (lambda_6 * lambda_5 * math.exp(-monotonicity_val))
            a = a - alpha * dL_da
            b = b - alpha * dL_db
        f_cap_cs = a*f_ps + b
        residuals = f_cs - f_cap_cs
        loss = (np.linalg.norm(residuals)/n) - lambda_hp*math.log(scale_hp * f_p_cov)
        
        if return_loss is True:
            return a, b, loss
        return a, b
    except:
        exc_type, exc_val, exc_tb = sys.exc_info()
        print "*** print_exception:"
        traceback.print_exception(exc_type, exc_val, exc_tb, file=sys.stdout)
        print a, b, f_cap_cs
        raise Exception('Error')

    
if __name__ == '__main__':
    import scipy.io as sio
    num_samples = int(sys.argv[1])
    data = sio.loadmat('am-only.mat')
    available_accs = data['available_accs']
    a1 = available_accs[1,:]
    a2 = available_accs[2,:]
    a, b = gradient_descent(a1[0:num_samples], a2[0:num_samples], lambda_1=1, lambda_3=1)
    a2_cap = a2*a + b
    print a, b
    print np.linalg.norm(a1 - a2_cap)
    print a1
    print a2_cap
