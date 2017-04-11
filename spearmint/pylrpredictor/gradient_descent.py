import sys
import traceback
import random
import numpy as np
import math

def gradient_descent(f_cs, f_ps,
                     alpha=0.01,
                     N=1000,
                     lambda_1=0,
                     lambda_2=100,
                     lambda_3=1,
                     lambda_4=1,
                     return_loss=False,
                     a_0=None,
                     b_0=None):
    
    if a_0 is None:
        a = random.random()
    else:
        a = a_0
    if b_0 is None:
        b = random.random() - 0.5
    else:
        b = b_0
    
    n = len(f_cs)
    try:
        assert(len(f_cs) == len(f_ps))
        for i in xrange(N):
            f_cap_cs = a*f_ps + b
            residuals = f_cs - f_cap_cs
            dL_da = -np.dot(residuals, f_ps)/n - (lambda_1 * lambda_2 * math.exp(-lambda_2 * a)) - (lambda_3 * (1.0 - a) / math.exp(lambda_4 * n))
            dL_db = -np.sum(residuals)/n
            a = a - alpha * dL_da
            b = b - alpha * dL_db
        f_cap_cs = a*f_ps + b
        residuals = f_cs - f_cap_cs
        loss = np.linalg.norm(residuals)/n
        
        if return_loss is True:
            return a, b, loss
        return a, b
    except:
        print a, b, f_cap_cs
        exc_type, exc_val, exc_tb = sys.exc_info()
        print "*** print_exception:"
        traceback.print_exception(exc_type, exc_val, exc_tb, file=sys.stdout)
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
