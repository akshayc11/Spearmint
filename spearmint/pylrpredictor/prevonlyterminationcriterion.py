import sys

from abc import abstractmethod
#from modelfactory import setup_prevonly_model_combination
from curvefunctions import all_models
import numpy as np
from pprint import pprint
from scipy.stats import norm
from gradient_descent import gradient_descent

IMPROVEMENT_PROB_THRESHOLD = 0.05
PREDICTIVE_STD_THRESHOLD = 0.005




class TerminationCriterion(object):
    """
    Base class for this form of early termination criterion.
    This uses multiple gradient descent starts to derive the
    parameters for matching previous runs to the current run
    with the constraint that the scaling factor is highly limited
    to be around 1 if the number of available samples for comparison
    is less, with the restriction being relaxed as more samples get
    available.
    """
    prob_x_greater_type = None
    xlim = None
    model = None
    has_fit =  False
    y_prev_list = None
    y_curr = None
    y_best = None
    a_b_losses = None

    def __init__(self, y_curr, xlim, prob_x_greater_type=None,
                 y_prev_list = [], n=100, predictive_std_threshold=PREDICTIVE_STD_THRESHOLD):
        """
        Constructor for the TerminationCriterion
        """

        self.prob_x_greater_type = prob_x_greater_type
        self.xlim = xlim
        self.y_prev_list = y_prev_list
        self.a_b_losses = []
        self.predictive_std_threshold=predictive_std_threshold
        self.has_fit = self.fit(y_curr, n)
        
    def get_prediction(self, xlim=None, thin=None):
        return self.predict(xlim=xlim)
    
    def fit(self, y_curr, n):
        self.y_curr = y_curr
        for y_idx in range(len(self.y_prev_list)):
            y_prev = self.y_prev_list[y_idx]
            a_b_losses = self.__get_gradients_and_losses(y_curr, y_prev, n=n)
            for a_b_loss in a_b_losses:
                self.a_b_losses.append(a_b_loss + (y_idx,))
        self.a_b_losses = sorted(self.a_b_losses, key=lambda x:x[2])[0:n]
        return True
    
    def predict(self, xlim=None, thin=None):
        r""" Predict the mean and standard deviation of the extrapolation using
        previous model information
        P(y_final | y_1:m; y_prevs)
        """
        if xlim is None:
            xlim = self.xlim
        
        y_caps = []
        if self.a_b_losses is None or len(self.a_b_losses) == 0:
            result = {'predictive_mean': 0.0,
                      'predictive_std': 1.0,
                      'found': False}
        else:
            for a_b_loss in self.a_b_losses:
                a, b, loss, y_idx = a_b_loss
                y_prev = self.y_prev_list[y_idx]
                y_cap = a*y_prev[xlim-1] + b
                if y_cap > 1.0 or y_cap < 0.0:
                    continue
                y_caps.append(y_cap)
            y_predict = np.mean(y_caps)
            y_std = np.std(y_caps)
            if y_predict >=0 and y_predict <= 1.0 and y_std >= 0.0:
                result = {"predictive_mean": y_predict,
                          "predictive_std": y_std,
                          "found": True}
            else:
                sys.stderr.write("y_predict is outside normal bounds: {} or incorrect std deviation: {}\n".format(y_predict, y_std))
                result = {"predictive_mean": y_predict,
                          "predictive_std": y_std,
                          "found": False}
        return result

    def posterior_prob_x_greater_than(self, y_best, xlim):
        '''
        posterior probability of predicted y at given time xlim is
        greater than a value y_best
        '''
        result = self.predict(xlim)
        if result['found'] == False:
            return 1.0
        else:
            predictive_mean = result['predictive_mean']
            predictive_std = result['predictive_std']
            return norm.cdf(y_best, loc=predictive_mean, scale=predictive_std)

    @abstractmethod
    def run(self, y_best, thin=None):
        """ The actual run function
        
        Abstract method to run the termination criterion check.
        
        Decorators:
            abstractmethod
        """
        pass

    def __get_gradients_and_losses(self, f_cs, f_ps, n=100):
        """
        This is a wrapper for the kind of gradient descent this
        termination criterion going to use. In this case, we will
        be using the default parameters.
        """
        min_len = min(len(f_cs), len(f_ps))
        f_c = f_cs[0:min_len]
        f_p = f_ps[0:min_len]
        a_b_losses = []
        for i in xrange(n):
            a_b_losses.append(gradient_descent(f_c, f_p, return_loss=True))
        
        return a_b_losses

class ConservativeTerminationCriterion(TerminationCriterion):

    def __init__(self, y_curr, xlim, prob_x_greater_type=None,
                 y_prev_list = [], n=100, predictive_std_threshold=PREDICTIVE_STD_THRESHOLD):
        super(ConservativeTerminationCriterion, self).__init__(
            y_curr, xlim, prob_x_greater_type,
            y_prev_list=y_prev_list, n=n)
    
    def run(self, y_best, threshold=IMPROVEMENT_PROB_THRESHOLD):
        """
        Run method for the conservative termination criterion
        """
        if y_curr is None or len(y_curr)==0:
            sys.stderr.write('No y_s done yet\n')
            result = {'predictive_mean': None,
                      'predictive_std': None,
                      'found': False,
                      'prob_gt_ybest_xlast': 0,
                      'terminate': False}
        else:
            y_c_best = np.max(y_list)
            if y_c_best > y_best:
                # let current build run to termination
                sys.stderr.write('Already exceeding previous best. Proceed to termination\n')
                result = {'predicitve_mean': y_c_best,
                          'predictive_std': 0.0,
                          'found': True,
                          'prob_gt_ybest_xlast': 1.0,
                          'terminate': False}
            else:
                if self.has_fit == False:
                    sys.stderr.write('Failed in fitting. Let training proceed in any case.\n')
                    result = {'predictive_mean': y_c_best,
                              'predictive_std': 0.0,
                              'found': False,
                              'prob_gt_ybest_xlast': 0,
                              'terminate': False}
                else:
                    prob_gt_ybest_xlast = self.posterior_prob_x_greater_than(self.xlim, y_best)
                    sys.stderr.write('P(y>y_best) = {}\n'.format(prob_gt_ybest_xlast))
                    res = self.predict()
                    predictive_mean = res['predictive_mean']
                    predictive_std = res['predictive_std']
                    found = res['found']
                    if prob_gt_ybest_xlast < threshold:
                        if predictive_std_threshold is None:
                            result = {'predictive_mean': predictive_mean,
                                      'predictive_std': predictive_std,
                                      'found': found,
                                      'prob_gt_ybest_xlast':
                                      prob_gt_ybest_xlast,
                                      'terminate': found}
                        else:
                            sys.stderr.write("std_predictive_threshold is set. Checking the predictive_std first\n")
                            sys.stderr.write("predictive_std: {}\n".format(
                                predictive_std))
                            if predictive_std < self.predictive_std_threshold:
                                sys.stderr.write("Predicting")
                                result = {'predictive_mean': predictive_mean,
                                          'predictive_std': predictive_std,
                                          'found': found,
                                          'prob_gt_ybest_xlast':
                                              prob_gt_ybest_xlast,
                                          'terminate': found}
                            else:
                                print "Continue Trainng\n"
                                result = {'predictive_mean': predictive_mean,
                                          'predictive_std': predictive_std,
                                          'found': found,
                                          'prob_gt_ybest_xlast':
                                              prob_gt_ybest_xlast,
                                          'terminate': False}
                    else:
                        print "Continue Training\n"
                        result = {'predictive_mean': predictive_mean,
                                  'predictive_std': predictive_std,
                                  'found': found,
                                  'prob_gt_ybest_xlast': prob_gt_ybest_xlast,
                                  'terminate': False}
        pprint(result)
        return result
