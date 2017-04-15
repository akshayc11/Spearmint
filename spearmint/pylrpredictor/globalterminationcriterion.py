"""
Early Criterion termination file.

This file contains the classes required to implement the early stopping
criterion

"""
# TODO: remove num_cut from xlim!


# from caffe.proto import caffe_pb2
# import google
# from google.protobuf import text_format

import sys

from abc import abstractmethod
from modelfactory import setup_model_combination
from curvefunctions import all_models
import numpy as np
from pprint import pprint

IMPROVEMENT_PROB_THRESHOLD = 0.05
PREDICTIVE_STD_THRESHOLD = 0.005

PREDICTION_THINNING = 10
NTHREADS = 4


def cut_beginning(y, threshold=0.05, look_ahead=5):
    """
    Cut the first few elements from the list of y_best.

    We start at a point where we are bigger than the initial value for
    look_ahead steps
    """
    if len(y) < look_ahead:
        return y
    num_cut = 0
    for idx in range(len(y) - look_ahead):
        start_here = True
        for idx_ahead in range(idx, idx + look_ahead):
            if not (y[idx_ahead] - y[0] > threshold):
                start_here = False
        if start_here:
            num_cut = idx
            break
    return y[num_cut:]


# def get_xlim():
#     assert os.path.exists("caffenet_solver.prototxt")
#     solver = caffe_pb2.SolverParameter()
#     solver_txt = open("caffenet_solver.prototxt").read()
#     try:
#         google.protobuf.text_format.Merge(solver_txt, solver)
#     except Exception as e:
#         #this may happen if fields are added. However everything else should
#          be parse
#         #hence, it's ok to be ignored
#         print "error parsing solver: ", str(e)
#     assert solver.max_iter > 0
#     assert solver.test_interval > 0
#     return solver.max_iter / float(solver.test_interval)


class TerminationCriterion(object):
    """Base class for early termination criterion check.

    This is the base class of the early termination criterion checks. It sets
    up the MCMC Model combination used for the extrapolation, and implements
    the predict method which will be used in run time.

    Variables:
        prob_x_greater_type {string} -- type of probability to use for
                                        termination
        xlim {int} -- maximum number of validation steps
        model {MCMCCurveModelCombination} -- The model object that is used for
                                             extrapolation.
    """

    prob_x_greater_type = None
    xlim = None
    model = None
    has_fit = False
    y_prev_list = None
    y_curr = None
    y_best = None

    def __init__(self,  xlim, nthreads, prob_x_greater_type,
                 recency_weighting=True, models=None,
                 y_prev_list=[],
                 const_param_models=['prev_linear']):
        """Constructor for TerminationCriterion class.

        The constructor for the TerminationCriterion class.
        Specifies the number of threads to be used in the model for MCMC and
        prediction.

        Arguments:
            xlim {int} -- total number of validations that are going to be run.
            nthreads {int} -- number of threads to be used for MCMC.
            prob_x_greater_type {string} -- whether a single sample or mean of
                                            samples are to be used for
                                            prediction.

        Keyword Arguments:
            recency_weighting {bool} -- [description] (default: {True})
        """
        # """
        # This is an updated version of the termination criterion check.
        # Here, it supersedes teh existing termination criterion check by
        # returning not only the expected value of the model, but also the
        # uncertainty from the model at a given time.
        # Assumption: This is being run from the correct base folder of
        # operation.
        # """
        self.prob_x_greater_type = prob_x_greater_type
        sys.stderr.write('prob_x_greater_type:{}\n'.format(
            self.prob_x_greater_type))

        # The models we will be using:
        if models is None:
            self.models = ["vap", "ilog2", "weibull", "pow3", "pow4", "loglog_linear",
                           "mmf", "janoschek", "dr_hill_zero_background", "log_power",
                           "exp4"]
        else:
            sys.stderr.write('{}\n'.format(models))
            for model in models:
                if not model in all_models:
                    raise Exception('Model {} not supported'.format(model))
            self.models = models
        # This function determines the max number of validation runs for the
        # given training
        self.xlim = xlim
        self.y_prev_list = y_prev_list
        self.const_param_list = []
        for y_prev in self.y_prev_list:
            self.const_param_list.append({'f_prev': np.array(list(y_prev))})
        sys.stderr.write("xlim:{}\n".format(self.xlim))
        self.model = setup_model_combination(
            xlim=xlim,
            models=self.models,
            const_param_list=self.const_param_list,
            const_param_models=const_param_models,
            recency_weighting=recency_weighting,
            nthreads=nthreads)

    @abstractmethod
    def run(self, y_curr, y_best, thin=PREDICTION_THINNING):
        """The actual run function.

        Abstract method to run the termination criterion check.

        Decorators:
            abstractmethod

        Arguments:
            y_curr {list} -- list of floats describing result at each
                             validation step
            y_best {float} -- best available validation accuracy seen so far

        Keyword Arguments:
            thin {int} -- number of steps between MCMC iterations to sample
                          (default: {PREDICTION_THINNING})

        Returns:
            [dict] -- dictionary of the predictive mean, predictive standard
                      deviation, whether they are reasonable, probability of
                      prediction being higher than y_best, and the decision to
                      terminate
        """
        pass
    
    def fit(self, y_curr, thin=PREDICTION_THINNING):
        x = np.asarray(range(1, len(y_curr)+1))
        self.has_fit = self.model.fit(x, y_curr)

    def get_prediction(self, xlim=None, thin=PREDICTION_THINNING):
        """
        Get the prediction for a given y_curr
        """
        if xlim is None:
            xlim = self.xlim
        if not self.has_fit:
            return "FAIL"
        return self.predict(xlim=xlim)

    def predict(self, xlim=None, thin=PREDICTION_THINNING):
        r"""Predict the mean and standard deviation of the extrapolation.

        Predicts the mean and standard deviation of the final accuracy given
        the model: P(y_final | y_{1:m}; \theta)

        Keyword Arguments:
            thin {int} --  (default: {PREDICTION_THINNING})

        Returns:
            [dict] -- dictionary of predictive_mean, predictive_std and whether
                      the two are reasonable and within bounds.
        """
        # We are mostly unlikely to improve.
        if xlim is None:
            xlim = self.xlim
        y_predict = self.model.predict(xlim, thin=thin)
        y_std = self.model.predictive_std(xlim, thin=thin)
        if y_predict >= 0. and y_predict <= 1.0 and y_std >= 0:
            result = {"predictive_mean": y_predict,
                      "predictive_std": y_std,
                      "found": True}
        else:
            sys.stderr.write("y_predict outside normal bounds:{} \
            or incorrect std deviation: {}\n".format(y_predict, y_std))
            result = {"predictive_mean": y_predict,
                      "predictive_std": y_std,
                      "found": False}
        return result


class OptimisticTerminationCriterion(TerminationCriterion):
    """The Optimistic Termination Criterion class.

    This class is the implementation of an optimistic form of termination that
    places more confidence on the predictive ability of the model. If the
    model's standard deviation of prediction falls below the threshold, then
    the result from the model is considered as truth. If not, then the
    probability that the system will outperform the best result seen so far
    is considered.

    Extends:
        TerminationCriterion

    Variables:
        predictive_std_threshold {float} -- The threshold below which the model
        is considered as ground truth.
    """

    predictive_std_threshold = None

    def __init__(self, xlim, nthreads, prob_x_greater_type,
                 predictive_std_threshold=PREDICTIVE_STD_THRESHOLD,
                 models=None):
        """Constructor for OptimisticTerminiationCriterion.

        The Constructor for the OptimisticTerminationCriterion class.

        Arguments:
            xlim {int} -- max number of validation results from a run
            nthreads {int} -- number of threads to be used for MCMC
            prob_x_greater_type {str} -- whether mean of samples or single
                                         sample is to be used.

        Keyword Arguments:
            predictive_std_threshold {float} -- the threshold below which model
                                                is considered accurate
                                                (default:
                                                  {PREDICTIVE_STD_THRESHOLD})
        """
        assert predictive_std_threshold > 0

        super(OptimisticTerminationCriterion,
              self).__init__(xlim, nthreads, prob_x_greater_type, models=models)
        self.predictive_std_threshold = predictive_std_threshold

    def run(self, y_curr, y_best, thin=PREDICTION_THINNING,
            threshold=IMPROVEMENT_PROB_THRESHOLD):
        """Run method for OptimisticTerminationCriterion class.

        The actual implementation of the termination criterion.

        Arguments:
            y_curr {list} -- all validation results seen so far for given run.
            y_best {float} -- best available validation seen so far.

        Keyword Arguments:
            thin {int} -- skips between sampling (default:
                                                    {PREDICTION_THINNING})
            threshold {float} -- threshold for probability that given run will
                                 exceed the best seen validation result
                                 (default: {IMPROVEMENT_PROB_THRESHOLD})

        Returns:
            [dict] -- containing predictive mean, predictive std deviation, if
                      the two are reasonable, probability of current run
                      exceeding best available value so far, and decision to
                      terminate.
        """
        '''
        This function builds the extrapolation model using y_curr and y_best,
        and based on the prediction from the extrapolation model, determines
        when to terminate.
        '''
        print "List: ", y_curr
        if len(y_curr) == 0:
            sys.stderr.write("No y_s done yet\n")
            result = {"predictive_mean": None,
                      "predictive_std": None,
                      "found": False,
                      'prob_gt_ybest_xlast': 0.0,
                      "terminate": False}
        else:
            y_curr_best = np.max(y_curr)
            if y_curr_best > y_best:
                sys.stderr.write("Already exceeding best. Proceed\n")
                result = {"predictive_mean": y_curr_best,
                          "predictive_std": 0.0,
                          "found": True,
                          'prob_gt_ybest_xlast': 1.0,
                          "terminate": False}
            else:
                y = y_curr
                # y = cut_beginning(y_curr)
                x = np.asarray(range(1, len(y) + 1))
                if not self.model.fit(x, y):
                    sys.stderr.write('Failed in fitting, Let the training proceed \
                        in any case\n')
                    result = {'predictive_mean': y_curr_best,
                              'predictive_std': 0,
                              'found': False,
                              'prob_gt_ybest_xlast': 0,
                              'terminate': False}

                res = self.predict(thin=thin)
                predictive_mean = res['predictive_mean']
                predictive_std = res['predictive_std']
                found = res['found']
                if predictive_std < self.predictive_std_threshold:
                    # The model is pretty sure about it's prediction
                    result = {'predictive_mean': predictive_mean,
                              'predictive_std': predictive_std,
                              'found': found,
                              'prob_gt_ybest_xlast': 0,
                              'terminate': found}

                elif y_best is not None:
                    sys.stderr.write("predictive_std is high. Checking probability \
                        of going higher than ybest\n")

                    if self.prob_x_greater_type == \
                            'posterior_prob_x_greater_than':
                        prob_gt_ybest_xlast = \
                            self.model.posterior_prob_x_greater_than(
                                self.xlim,
                                y_best,
                                thin=thin)
                    else:
                        prob_gt_ybest_xlast = \
                            self.model.posterior_mean_prob_x_greater_than(
                                self.xlim,
                                y_best,
                                thin=thin)

                    sys.stderr.write("P(y > y_best) = {}\n".format(
                        prob_gt_ybest_xlast))

                    if prob_gt_ybest_xlast < threshold:
                        # Below the threshold. Send the termination signals
                        result = {'predictive_mean': predictive_mean,
                                  'predictive_std': predictive_std,
                                  'found': found,
                                  'prob_gt_ybest_xlast': prob_gt_ybest_xlast,
                                  'terminate': found}
                    else:
                        sys.stderr.write("Continue Training\n")
                        result = {'predictive_mean': predictive_mean,
                                  'predictive_std': predictive_std,
                                  'found': found,
                                  'prob_gt_ybest_xlast': prob_gt_ybest_xlast,
                                  'terminate': False}

                else:
                    sys.stderr.write("neither low std, nor ybest present\n")
                    result = {'predictive_mean': predictive_mean,
                              'predictive_std': predictive_std,
                              'found': found,
                              'prob_gt_ybest_xlast': 0,
                              'terminate': False}

        pprint(result)
        return result


class ConservativeTerminationCriterion(TerminationCriterion):
    """The Conservative Termination Criterion class.

    This class is the implementation of an conservative form of termination.
    If the prediction probability is lesser than a threshold, we allow training
    to continue. Else, we check if the standard deviation of the model's
    prediction was below an optional threshold. If so, then we terminate the
    run. If not, we let the training proceed.

    Extends:
        TerminationCriterion

    Variables:
        predictive_std_threshold {float} -- The threshold below which the model
        is considered as ground truth.
    """

    predictive_std_threshold = None

    def __init__(self, xlim, nthreads, prob_x_greater_type,
                 predictive_std_threshold=None, models=None):
        """Constructor for the ConservativeTerminationCriterion class.

        The constructor for the ConservativeTerminationCriterion class.

        Arguments:
            xlim {int} -- max number of validation results from a run
            nthreads {int} -- number of threads to be used for MCMC
            prob_x_greater_type {str} -- whether mean of samples or single
                                         sample is to be used.

        Keyword Arguments:
            predictive_std_threshold {float} -- the threshold below which model
                                                is considered accurate
                                                (default:
                                                  {PREDICTIVE_STD_THRESHOLD})
        """
        self.predictive_std_threshold = predictive_std_threshold
        super(ConservativeTerminationCriterion,
              self).__init__(xlim, nthreads, prob_x_greater_type, models=models)

    def run(self, y_curr, y_best, thin=PREDICTION_THINNING,
            threshold=IMPROVEMENT_PROB_THRESHOLD):
        """Run method for the ConservativeTerminationCriterion class.

        The actual implementation of the termination criterion.

        Arguments:
            y_curr {list} -- all validation results seen so far for given run.
            y_best {float} -- best available validation seen so far.

        Keyword Arguments:
            thin {int} -- skips between sampling (default:
                                                    {PREDICTION_THINNING})
            threshold {float} -- threshold for probability that given run will
                                 exceed the best seen validation result
                                 (default: {IMPROVEMENT_PROB_THRESHOLD})

        Returns:
            [dict] -- containing predictive mean, predictive std deviation, if
                      the two are reasonable, probability of current run
                      exceeding best available value so far, and decision to
                      terminate.
        """
        print "List: ", y_curr
        if len(y_curr) == 0:
            sys.stderr.write("No y_s done yet\n")
            result = {"predictive_mean": None,
                      "predictive_std": None,
                      "found": False,
                      'prob_gt_ybest_xlast': 0,
                      "terminate": False}
        else:
            y_curr_best = np.max(y_curr)
            if y_curr_best > y_best:
                sys.stderr.write("Already exceeding best. Proceed\n")
                result = {"predictive_mean": y_curr_best,
                          "predictive_std": 0,
                          "found": True,
                          'prob_gt_ybest_xlast': 1.0,
                          "terminate": False}
            else:
                #y = cut_beginning(y_curr)
                y = y_curr
                x = np.asarray(range(1, len(y) + 1))
                result = None
                if not self.model.fit(x, y):
                    sys.stderr.write('Failed in fitting, Let the training proceed \
                        in any case\n')
                    result = {'predictive_mean': y_curr_best,
                              'predictive_std': 0,
                              'found': False,
                              'prob_gt_ybest_xlast': 0,
                              'terminate': False}
                else:
                    if self.prob_x_greater_type == \
                            'posterior_prob_x_greater_than':
                        prob_gt_ybest_xlast = \
                            self.model.posterior_prob_x_greater_than(
                                self.xlim,
                                y_best,
                                thin=thin)
                    else:
                        prob_gt_ybest_xlast = \
                            self.model.posterior_mean_prob_x_greater_than(
                                self.xlim,
                                y_best,
                                thin=thin)

                    sys.stderr.write("P(y > y_best) = {}\n".format(
                        prob_gt_ybest_xlast))

                    res = self.predict(thin=thin)
                    predictive_mean = res['predictive_mean']
                    predictive_std = res['predictive_std']
                    found = res['found']

                    if prob_gt_ybest_xlast < threshold:
                        # Below the threshold. Send the termination signals
                        if self.predictive_std_threshold is None:
                            result = {'predictive_mean': predictive_mean,
                                      'predictive_std': predictive_std,
                                      'found': found,
                                      'prob_gt_ybest_xlast':
                                          prob_gt_ybest_xlast,
                                      'terminate': found}
                        else:
                            sys.stderr.write("std_predictive_threshold is set. \
                                Checking the predictive_std first\n")
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
