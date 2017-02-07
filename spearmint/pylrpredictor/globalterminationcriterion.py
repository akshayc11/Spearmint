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
    y_lists = None
    curr_y_list = None
    y_best = None
    models = None

    def __init__(self, y_lists, curr_y_list, y_best, xlim, nthreads,
                 prob_x_greater_type, recency_weighting=True, models=None):
        """Constructor for TerminationCriterion class.

        The constructor for the TerminationCriterion class.
        Specifies the number of threads to be used in the model for MCMC and
        prediction.

        Arguments:
            y_lists {list of list} -- list of list of floats describing result
                             at each validation step for previous runs
            curr_y_list {list} -- list of floats describing results of each
                                    validation step from current run.
            y_best {float} -- best available validation accuracy seen so far
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
            models = ["vap", "ilog2", "weibull", "pow3", "pow4", "loglog_linear",
                      "mmf", "janoschek", "dr_hill_zero_background", "log_power",
                      "exp4"]
        else:
            for model in models:
                assert model in all_models, "model {} not supported".format(model)
        self.models = models
        # This function determines the max number of validation runs for the
        # given training
        self.xlim = xlim
        sys.stderr.write("xlim:{}\n".format(self.xlim))
        self.curr_y_list = curr_y_list

        num_ys = len(curr_y_list)
        offsetted_y_lists = []
        for i in range(len(y_lists)):
            y_list = y_lists[i]
            len_i = len(y_list)
            max_len_for_offset = min(len_i, num_ys)
            if max_len_for_offset == 0:
                offset = 0.0
            else:
                offset_curr_list = curr_y_list[0:max_len_for_offset]
                offset_y_list = y_list[0:max_len_for_offset]
                offset = np.mean(np.array(offset_curr_list) -
                                 np.array(offset_y_list))
            offsetted_y_lists.append(y_list + offset)
        # Add the current list too to build a model for.
        offsetted_y_lists.append(curr_y_list)

        self.y_lists = offsetted_y_lists

        self.y_best = y_best

        self.models = [setup_model_combination(
            xlim=xlim,
            models=self.models,
            recency_weighting=recency_weighting,
            nthreads=nthreads) for offseted_y_list in self.y_lists]

    @abstractmethod
    def run(self, thin=PREDICTION_THINNING):
        """The actual run function.

        Abstract method to run the termination criterion check.

        Decorators:
            abstractmethod


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
    
    def fit(self):
        fits = []
        for idx in xrange(len(self.models)):
            y = self.y_lists[idx]
            # cut_beginning(y_list)
            x = np.asarray(range(1, len(y) + 1))
            result = None
            fits.append(self.models[idx].fit(x, y))
        self.fits = np.array(fits)
        sys.stderr.write('Fits: {}\n'.format(self.fits))
    
    def get_prediction(self, xlim=None, thin=PREDICTION_THINNING):
        if xlim is None:
            xlim = self.xlim
        if not np.any(self.fits):
            return "FAIL"
        return self.predict(self.fits, xlim=xlim, thin=thin)
        
    def predict(self, fits, xlim=None, thin=PREDICTION_THINNING):
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

        y_predicts = []
        y_stds = []
        for i in range(len(fits)):
            if fits[i] is True:
                y_predicts.append(self.models[i].predict(self.xlim, thin=thin))
                y_stds.append(self.models[i].predictive_std(self.xlim,
                                                            thin=thin))

        sys.stderr.write("y_predicts: {}\n".format(y_predicts))
        sys.stderr.write("y_stds: {}\n".format(y_stds))
        y_predict = np.mean(np.ma.masked_invalid(np.asarray(y_predicts)))
        y_std = np.mean(np.ma.masked_invalid(np.asarray(y_stds)))

        if y_predict >= 0. and y_predict <= 1.0 and y_std >= 0:
            result = {"predictive_mean": y_predict,
                      "predictive_std": y_std,
                      "found": True}
        else:
            sys.stderr.write(
                "y_predict outside normal bounds:{} or incorrect std deviation: {}\n".format(
                    y_predict, y_std))
            result = {"predictive_mean": y_predict,
                      "predictive_std": y_std,
                      "found": False}
        return result


class NewTerminationCriterion(object):
    """Base class for early termination criterion check.

    This is the base class of the early termination criterion checks. It sets
    up the MCMC Model combination used for the extrapolation, and implements
    the predict method which will be used in run time.
    Any offsetted runs going over 1.0 or below 0.0 are discarded.

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
    y_lists = None
    curr_y_list = None
    y_best = None
    models = None

    def __init__(self, y_lists, curr_y_list, y_best, xlim, nthreads,
                 prob_x_greater_type, recency_weighting=True, models=None):
        """Constructor for TerminationCriterion class.

        The constructor for the TerminationCriterion class.
        Specifies the number of threads to be used in the model for MCMC and
        prediction.

        Arguments:
            y_lists {list of list} -- list of list of floats describing result
                             at each validation step for previous runs
            curr_y_list {list} -- list of floats describing results of each
                                    validation step from current run.
            y_best {float} -- best available validation accuracy seen so far
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
            models = ["vap", "ilog2", "weibull", "pow3", "pow4", "loglog_linear",
                      "mmf", "janoschek", "dr_hill_zero_background", "log_power",
                      "exp4"]
        else:
            for model in models:
                assert model in all_models, "model {} not supported".format(model)
        self.models = models
        # This function determines the max number of validation runs for the
        # given training
        self.xlim = xlim
        sys.stderr.write("xlim:{}\n".format(self.xlim))
        self.curr_y_list = curr_y_list

        num_ys = len(curr_y_list)
        offsetted_y_lists = []
        for i in range(len(y_lists)):
            y_list = y_lists[i]
            len_i = len(y_list)
            max_len_for_offset = min(len_i, num_ys)
            if max_len_for_offset == 0:
                offset = 0.0
            else:
                offset_curr_list = curr_y_list[0:max_len_for_offset]
                offset_y_list = y_list[0:max_len_for_offset]
                offset = np.mean(np.array(offset_curr_list) -
                                 np.array(offset_y_list))
            new_list = y_list + offset
            sys.stderr.write('data:{} offset: {} \nold: {}\nnew: {}\n'.format(
                i, offset, y_list, new_list))
            if np.any(new_list > 1.0) or np.any(new_list < 0.0):
                sys.stderr.write(
                    "Discarding Run {} from models due to offset causing violation.\n".format(i))
                continue
            offsetted_y_lists.append(new_list)
        # Add the current list too to build a model for.
        offsetted_y_lists.append(curr_y_list)

        self.y_lists = offsetted_y_lists

        self.y_best = y_best

        self.models = [setup_model_combination(
            xlim=xlim,
            models=self.models,
            recency_weighting=recency_weighting,
            nthreads=nthreads) for offseted_y_list in self.y_lists]

    @abstractmethod
    def run(self, thin=PREDICTION_THINNING):
        """The actual run function.

        Abstract method to run the termination criterion check.

        Decorators:
            abstractmethod


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
    
    def fit(self):
        fits = []
        for idx in xrange(len(self.models)):
            y = self.y_lists[idx]
            # cut_beginning(y_list)
            x = np.asarray(range(1, len(y) + 1))
            fits.append(self.models[idx].fit(x, y))
        self.fits = fits
    
    def get_prediction(self, xlim=None, thin=PREDICTION_THINNING):
        if xlim is None:
            xlim = self.xlim
        if not np.any(self.fits):
            return "FAIL"
        return self.predict(self.fits, xlim=xlim, thin=thin)
        
    def predict(self, fits, xlim=None, thin=PREDICTION_THINNING):
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

        sys.stderr.write('In predict Fits: {}\n'.format(self.fits))
        y_predicts = []
        y_stds = []
        if xlim is None:
            xlim = self.xlim
        sys.stderr.write('xlim: {} {}\n'.format(xlim, fits[0] is True))
        for i in range(len(fits)):
            if fits[i] is True:
                sys.stderr.write('model: {}\n'.format(i))
                y_predicts.append(self.models[i].predict(xlim, thin=thin))
                y_stds.append(self.models[i].predictive_std(xlim,
                                                            thin=thin))

        sys.stderr.write("y_predicts: {}\n".format(y_predicts))
        sys.stderr.write("y_stds: {}\n".format(y_stds))
        y_predict = np.mean(np.ma.masked_invalid(np.asarray(y_predicts)))
        y_std = np.mean(np.ma.masked_invalid(np.asarray(y_stds)))

        if y_predict >= 0. and y_predict <= 1.0 and y_std >= 0:
            result = {"predictive_mean": y_predict,
                      "predictive_std": y_std,
                      "found": True}
        else:
            sys.stderr.write(
                "y_predict outside normal bounds:{} or incorrect std deviation: {}\n".format(
                    y_predict, y_std))
            result = {"predictive_mean": y_predict,
                      "predictive_std": y_std,
                      "found": False}
        return result




class NewTerminationCriterionV2(object):
    """NewBase class for early termination criterion check.

    This is the base class of the early termination criterion checks. It sets
    up the MCMC Model combination used for the extrapolation, and implements
    the predict method which will be used in run time. Any offsetted runs that
    have extrapolations going over 1.0 or below 0.0 are discarded.

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
    y_lists = None
    curr_y_list = None
    y_best = None

    def __init__(self, y_lists, curr_y_list, y_best, xlim, nthreads,
                 prob_x_greater_type, recency_weighting=True):
        """Constructor for TerminationCriterion class.

        The constructor for the TerminationCriterion class.
        Specifies the number of threads to be used in the model for MCMC and
        prediction.

        Arguments:
            y_lists {list of list} -- list of list of floats describing result
                             at each validation step for previous runs
            curr_y_list {list} -- list of floats describing results of each
                                    validation step from current run.
            y_best {float} -- best available validation accuracy seen so far
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
        models = ["vap", "ilog2", "weibull", "pow3", "pow4", "loglog_linear",
                  "mmf", "janoschek", "dr_hill_zero_background", "log_power",
                  "exp4"]

        # This function determines the max number of validation runs for the
        # given training
        self.xlim = xlim
        sys.stderr.write("xlim:{}\n".format(self.xlim))
        self.curr_y_list = curr_y_list

        num_ys = len(curr_y_list)
        offsetted_y_lists = []
        for i in range(len(y_lists)):
            y_list = y_lists[i]
            if len(y_list) < self.xlim / 2:
                sys.stderr.write("disregarding run since less than half of validation runs were done")
                continue
            len_i = len(y_list)
            max_len_for_offset = min(len_i, num_ys)
            if max_len_for_offset == 0:
                offset = 0.0
            else:
                offset_curr_list = curr_y_list[0:max_len_for_offset]
                offset_y_list = y_list[0:max_len_for_offset]
                offset = np.mean(np.array(offset_curr_list) -
                                 np.array(offset_y_list))
            new_list = y_list + offset
            sys.stderr.write('data:{} offset: {} \nold: {}\nnew: {}\n'.format(
                i, offset, y_list, new_list))
            if np.any(new_list > 1.0) or np.any(new_list < 0.0):
                sys.stderr.write(
                    "Discarding Run {} from models due to offset causing violation.\n".format(i))
                continue
            offsetted_y_lists.append(new_list)
        # Add the current list too to build a model for.
        offsetted_y_lists.append(curr_y_list)

        self.y_lists = offsetted_y_lists

        self.y_best = y_best

        self.models = [setup_model_combination(
            xlim=xlim,
            models=models,
            recency_weighting=recency_weighting,
            nthreads=nthreads) for offseted_y_list in self.y_lists]

    @abstractmethod
    def run(self, thin=PREDICTION_THINNING):
        """The actual run function.

        Abstract method to run the termination criterion check.

        Decorators:
            abstractmethod


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

    def predict(self, fits, thin=PREDICTION_THINNING):
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

        y_predicts = []
        y_stds = []
        for i in range(len(fits)):
            if fits[i] is True:
                y_predicts.append(self.models[i].predict(self.xlim, thin=thin))
                y_stds.append(self.models[i].predictive_std(self.xlim,
                                                            thin=thin))

        sys.stderr.write("y_predicts: {}".format(y_predicts))
        sys.stderr.write("y_stds:".format(y_stds))
        y_predict = np.mean(np.ma.masked_invalid(np.asarray(y_predicts)))
        y_std = np.mean(np.ma.masked_invalid(np.asarray(y_stds)))

        if y_predict >= 0. and y_predict <= 1.0 and y_std >= 0:
            result = {"predictive_mean": y_predict,
                      "predictive_std": y_std,
                      "found": True}
        else:
            sys.stderr.write("y_predict outside normal bounds:{} or incorrect std deviation: {}\n".format(y_predict, y_std))
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

    def __init__(self, y_lists, curr_y_list, y_best,
                 xlim, nthreads, prob_x_greater_type,
                 predictive_std_threshold=PREDICTIVE_STD_THRESHOLD):
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
              self).__init__(y_lists, curr_y_list, y_best,
                             xlim, nthreads, prob_x_greater_type)
        self.predictive_std_threshold = predictive_std_threshold

    def run(self, thin=PREDICTION_THINNING,
            threshold=IMPROVEMENT_PROB_THRESHOLD):
        """Run method for OptimisticTerminationCriterion class.

        The actual implementation of the termination criterion.

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
        This function builds the extrapolation model using y_list and y_best,
        and based on the prediction from the extrapolation model, determines
        when to terminate.
        '''
        sys.stderr.write("Current List: {}\n".format(self.curr_y_list))
        if len(self.curr_y_list) == 0:
            sys.stderr.write("No y_s done yet\n")
            result = {"predictive_mean": None,
                      "predictive_std": None,
                      "found": False,
                      'prob_gt_ybest_xlast': 0.0,
                      "terminate": False}
        else:
            y_curr_best = np.max(self.curr_y_list)
            if y_curr_best > self.y_best:
                sys.stderr.write("Already exceeding best. Proceed\n")
                result = {"predictive_mean": y_curr_best,
                          "predictive_std": 0.0,
                          "found": True,
                          'prob_gt_ybest_xlast': 1.0,
                          "terminate": False}
            else:
                fits = []
                for idx in range(len(self.models)):
                    y = self.y_lists[idx]
                    model = self.models[idx]
                    x = np.asarray(range(1, len(y) + 1))
                    fits.append(model.fit(x, y))
                if not np.any(fits):
                    sys.stderr.write("Failed in fitting. Let the training proceed\n")
                    result = {'predictive_mean': y_curr_best,
                              'predictive_std': 0,
                              'found': False,
                              'prob_gt_ybest_xlast': 0,
                              'terminate': False}
                else:
                    res = self.predict(fits, thin=thin)
                    predictive_mean = res['predictive_mean']
                    predictive_std = res['predictive_std']
                    found = res['found']
                    if predictive_std < self.predictive_std_threshold:
                        sys.stderr.write('model is pretty sure of prediction.\n')
                        result = {'predictive_mean': predictive_mean,
                                  'predictive_std': predictive_std,
                                  'found': found,
                                  'prob_gt_ybest_xlast': 0,
                                  'terminate': found}
                    elif self.y_best is not None:
                        sys.stderr.write('Predicitve std is high. Checking probability of going higher than y_best')

                        probs = []
                        if self.prob_x_greater_type == \
                                'posterior_prob_x_greater_than':
                            for i in range(len(self.models)):
                                model = self.models[i]
                                if fits[i] is True:
                                    probs.append(
                                        model.posterior_prob_x_greater_than(
                                            self.xlim,
                                            self.y_best, thin=thin))
                        else:
                            for i in range(len(self.models)):
                                if fits[i] is True:
                                    model = self.models[i]
                                    probs.append(
                                        model.posterior_prob_x_greater_than(
                                            self.xlim,
                                            self.y_best, thin=thin))
                        # Not sure if averaging probabilities makes sense
                        prob_gt_ybest_xlast = np.mean(probs)
                        sys.stderr.write('P(y>y_best) = {}\n'.format(
                            prob_gt_ybest_xlast))
                        if prob_gt_ybest_xlast < threshold:
                            sys.stderr.write('Below the threshold. Send the termination signal if found.\n')
                            result = {
                                'predictive_mean': predictive_mean,
                                'predictive_std': predictive_std,
                                'found': found,
                                'prob_gt_ybest_xlast': prob_gt_ybest_xlast,
                                'terminate': found}
                        else:
                            sys.stderr.write("Continue Training\n")
                            result = {
                                'predictive_mean': predictive_mean,
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

    def __init__(self, y_lists, curr_y_list, y_best,
                 xlim, nthreads, prob_x_greater_type,
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
              self).__init__(
                  y_lists, curr_y_list, y_best, xlim, nthreads,
                  prob_x_greater_type, models=models)

    def run(self, thin=PREDICTION_THINNING,
            threshold=IMPROVEMENT_PROB_THRESHOLD):
        """Run method for the ConservativeTerminationCriterion class.

        The actual implementation of the termination criterion.

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
        sys.stderr.write("Current List: {}\n".format(self.curr_y_list))
        if len(self.curr_y_list) == 0:
            sys.stderr.write("No y_s done yet\n")
            result = {"predictive_mean": None,
                      "predictive_std": None,
                      "found": False,
                      'prob_gt_ybest_xlast': 0,
                      "terminate": False}
        else:
            y_curr_best = np.max(self.curr_y_list)
            if y_curr_best > self.y_best:
                sys.stderr.write("Already exceeding best. Proceed\n")
                result = {"predictive_mean": y_curr_best,
                          "predictive_std": 0,
                          "found": True,
                          'prob_gt_ybest_xlast': 1.0,
                          "terminate": False}
            else:
                fits = []
                for idx in xrange(len(self.models)):
                    y = self.y_lists[idx]
                    # cut_beginning(y_list)
                    x = np.asarray(range(1, len(y) + 1))
                    result = None
                    fits.append(self.models[idx].fit(x, y))

                if not np.any(fits):
                    sys.stderr.write('Failed in fitting, Let the training proceed in any case\n')
                    result = {'predictive_mean': y_curr_best,
                              'predictive_std': 0,
                              'found': False,
                              'prob_gt_ybest_xlast': 0,
                              'terminate': False}
                else:
                    probs = []
                    if self.prob_x_greater_type == \
                            'posterior_prob_x_greater_than':
                        for i in range(len(self.models)):
                            if fits[i] is True:
                                model = self.models[i]
                                probs.append(
                                    model.posterior_prob_x_greater_than(
                                        self.xlim,
                                        self.y_best,
                                        thin=thin))
                    else:
                        for i in range(len(self.models)):
                            if fits[i] is True:
                                model = self.models[i]
                                probs.append(
                                    model.posterior_mean_prob_x_greater_than(
                                        self.xlim,
                                        self.y_best,
                                        thin=thin))

                    # Not sure if averaging probabilities makes sense:
                    prob_gt_ybest_xlast = np.mean(probs)

                    sys.stderr.write("P(y > y_best) = {}\n".format(prob_gt_ybest_xlast))

                    res = self.predict(fits, thin=thin)
                    predictive_mean = res['predictive_mean']
                    predictive_std = res['predictive_std']
                    found = res['found']

                    if prob_gt_ybest_xlast < threshold:
                        # Below the threshold. Send the termination signals
                        if self.predictive_std_threshold is None:
                            result = {
                                'predictive_mean': predictive_mean,
                                'predictive_std': predictive_std,
                                'found': found,
                                'prob_gt_ybest_xlast': prob_gt_ybest_xlast,
                                'terminate': found}
                        else:
                            sys.stderr.write(
                                "std_predictive_threshold is set. Checking the predictive_std first\n")
                            sys.stderr.write("predictive_std: {}\n".format(
                                predictive_std))
                            if predictive_std < self.predictive_std_threshold:
                                sys.stderr.write("Predicting..\n")
                                result = {'predictive_mean': predictive_mean,
                                          'predictive_std': predictive_std,
                                          'found': found,
                                          'prob_gt_ybest_xlast':
                                              prob_gt_ybest_xlast,
                                          'terminate': found}
                            else:
                                sys.stderr.write("Continue Training..\n")
                                result = {'predictive_mean': predictive_mean,
                                          'predictive_std': predictive_std,
                                          'found': found,
                                          'prob_gt_ybest_xlast':
                                              prob_gt_ybest_xlast,
                                          'terminate': False}
                    else:
                        sys.stderr.write("Continue Training..\n")
                        result = {'predictive_mean': predictive_mean,
                                  'predictive_std': predictive_std,
                                  'found': found,
                                  'prob_gt_ybest_xlast': prob_gt_ybest_xlast,
                                  'terminate': False}

        pprint(result)
        return result


class NewConservativeTerminationCriterionV1(TerminationCriterion):
    """New Conservative Termination Criterion.

    This Termination Criterion is a bit less conservative. Along with the
    standard conditions, it also looks at the distance between the
    mean + N standard deviations of the prediction to see if that number is
    within the threshold. If not, terminate.

    Extends:
        TerminationCriterion

    Variables:
        N: margin of standard deviation to consider when terminating
    """

    n = None
    predictive_std_threshold = None

    def __init__(self, y_lists, curr_y_list, y_best,
                 xlim, nthreads, prob_x_greater_type,
                 predictive_std_threshold=None, n=5):
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
        self.n = n
        super(NewConservativeTerminationCriterionV1,
              self).__init__(y_lists, curr_y_list, y_best,
                             xlim, nthreads, prob_x_greater_type)

    def run(self, thin=PREDICTION_THINNING,
            threshold=IMPROVEMENT_PROB_THRESHOLD):
        """Run method for the ConservativeTerminationCriterion class.

        The actual implementation of the termination criterion.

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
        sys.stderr.write("Current List: {}\n".format(self.curr_y_list))
        if len(self.curr_y_list) == 0:
            sys.stderr.write("No y_s done yet\n")
            result = {"predictive_mean": None,
                      "predictive_std": None,
                      "found": False,
                      'prob_gt_ybest_xlast': 0,
                      "terminate": False}
        else:
            y_curr_best = np.max(self.curr_y_list)
            if y_curr_best > self.y_best:
                sys.stderr.write("Already exceeding best. Proceed\n")
                result = {"predictive_mean": y_curr_best,
                          "predictive_std": 0,
                          "found": True,
                          'prob_gt_ybest_xlast': 1.0,
                          "terminate": False}
            else:
                fits = []
                for idx in xrange(len(self.models)):
                    y = self.y_lists[idx]
                    # cut_beginning(y_list)
                    x = np.asarray(range(1, len(y) + 1))
                    result = None
                    fits.append(self.models[idx].fit(x, y))

                if not np.any(fits):
                    sys.stderr.write('Failed in fitting, Let the training proceed in any case\n')
                    result = {'predictive_mean': y_curr_best,
                              'predictive_std': 0,
                              'found': False,
                              'prob_gt_ybest_xlast': 0,
                              'terminate': False}
                else:
                    probs = []
                    if self.prob_x_greater_type == \
                            'posterior_prob_x_greater_than':
                        for i in range(len(self.models)):
                            if fits[i] is True:
                                model = self.models[i]
                                probs.append(
                                    model.posterior_prob_x_greater_than(
                                        self.xlim,
                                        self.y_best,
                                        thin=thin))
                    else:
                        for i in range(len(self.models)):
                            if fits[i] is True:
                                model = self.models[i]
                                probs.append(
                                    model.posterior_mean_prob_x_greater_than(
                                        self.xlim,
                                        self.y_best,
                                        thin=thin))

                    # Not sure if averaging probabilities makes sense:
                    prob_gt_ybest_xlast = np.mean(probs)

                    sys.stderr.write("P(y > y_best) = {}\n".format(prob_gt_ybest_xlast))

                    res = self.predict(fits, thin=thin)
                    predictive_mean = res['predictive_mean']
                    predictive_std = res['predictive_std']
                    found = res['found']
                    if found is False:
                        result = {'predictive_mean': predictive_mean,
                                  'predictive_std': predictive_std,
                                  'found': False,
                                  'prob_gt_ybest_xlast': prob_gt_ybest_xlast,
                                  'terminate': False}
                    else:
                        # Condition if predictive mean + n*predictive_std is
                        # less than best result, terminate
                        upper_bound = predictive_mean + self.n * predictive_std
                        if upper_bound < self.y_best:
                            sys.stderr.write('predictive mean {} is more than {} standard deviations away from the y_best {}. Terminate.\n'.format(predictive_mean, self.y_best, self.n))
                            result = {'predictive_mean': predictive_mean,
                                      'predictive_std': predictive_std,
                                      'found': True,
                                      'prob_gt_ybest_xlast': 0.0,
                                      'terminate': True}
                        else:
                            if prob_gt_ybest_xlast < threshold:
                                # Below the threshold. Send the termination
                                # signals
                                if self.predictive_std_threshold is None:
                                    result = {
                                        'predictive_mean': predictive_mean,
                                        'predictive_std': predictive_std,
                                        'found': found,
                                        'prob_gt_ybest_xlast':
                                        prob_gt_ybest_xlast,
                                        'terminate': found}
                                else:
                                    sys.stderr.write(
                                        "std_predictive_threshold set. Checking the predictive_std first\n")
                                    sys.stderr.write(
                                        "predictive_std: {}\n".format(
                                            predictive_std))
                                    if (predictive_std <
                                            self.predictive_std_threshold):
                                        sys.stderr.write("Predicting..\n")
                                        result = {
                                            'predictive_mean': predictive_mean,
                                            'predictive_std': predictive_std,
                                            'found': found,
                                            'prob_gt_ybest_xlast':
                                            prob_gt_ybest_xlast,
                                            'terminate': found}
                                    else:
                                        sys.stderr.write("Continue Training..\n")
                                        result = {
                                            'predictive_mean': predictive_mean,
                                            'predictive_std': predictive_std,
                                            'found': found,
                                            'prob_gt_ybest_xlast':
                                            prob_gt_ybest_xlast,
                                            'terminate': False}
                            else:
                                sys.stderr.write("Continue Training..\n")
                                result = {
                                    'predictive_mean': predictive_mean,
                                    'predictive_std': predictive_std,
                                    'found': found,
                                    'prob_gt_ybest_xlast': prob_gt_ybest_xlast,
                                    'terminate': False}

        pprint(result)
        return result


class NewConservativeTerminationCriterionV2(NewTerminationCriterion):
    """The Conservative Termination Criterion class.

    This class is the implementation of an conservative form of termination.
    If the prediction probability is lesser than a threshold, we allow training
    to continue. Else, we check if the standard deviation of the model's
    prediction was below an optional threshold. If so, then we terminate the
    run. If not, we let the training proceed.

    Extends:
        NewTerminationCriterion

    Variables:
        predictive_std_threshold {float} -- The threshold below which the model
        is considered as ground truth.
    """

    predictive_std_threshold = None

    def __init__(self, y_lists, curr_y_list, y_best,
                 xlim, nthreads, prob_x_greater_type,
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
        super(NewConservativeTerminationCriterionV2,
              self).__init__(
                  y_lists, curr_y_list, y_best, xlim, nthreads, prob_x_greater_type, models=models)

    def run(self, thin=PREDICTION_THINNING,
            threshold=IMPROVEMENT_PROB_THRESHOLD):
        """Run method for the ConservativeTerminationCriterion class.

        The actual implementation of the termination criterion.

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
        sys.stderr.write("Current List: {}\n".format(self.curr_y_list))
        if len(self.curr_y_list) == 0:
            sys.stderr.write("No y_s done yet\n")
            result = {"predictive_mean": None,
                      "predictive_std": None,
                      "found": False,
                      'prob_gt_ybest_xlast': 0,
                      "terminate": False}
        else:
            y_curr_best = np.max(self.curr_y_list)
            if y_curr_best > self.y_best:
                sys.stderr.write("Already exceeding best. Proceed\n")
                result = {"predictive_mean": y_curr_best,
                          "predictive_std": 0,
                          "found": True,
                          'prob_gt_ybest_xlast': 1.0,
                          "terminate": False}
            else:
                fits = []
                for idx in xrange(len(self.models)):
                    y = self.y_lists[idx]
                    # cut_beginning(y_list)
                    x = np.asarray(range(1, len(y) + 1))
                    result = None
                    fits.append(self.models[idx].fit(x, y))

                if not np.any(fits):
                    sys.stderr.write('Failed in fitting, Let the training proceed in any case\n')
                    result = {'predictive_mean': y_curr_best,
                              'predictive_std': 0,
                              'found': False,
                              'prob_gt_ybest_xlast': 0,
                              'terminate': False}
                else:
                    probs = []
                    if self.prob_x_greater_type == \
                            'posterior_prob_x_greater_than':
                        for i in range(len(self.models)):
                            if fits[i] is True:
                                model = self.models[i]
                                probs.append(
                                    model.posterior_prob_x_greater_than(
                                        self.xlim,
                                        self.y_best,
                                        thin=thin))
                    else:
                        for i in range(len(self.models)):
                            if fits[i] is True:
                                model = self.models[i]
                                probs.append(
                                    model.posterior_mean_prob_x_greater_than(
                                        self.xlim,
                                        self.y_best,
                                        thin=thin))

                    # Not sure if averaging probabilities makes sense:
                    prob_gt_ybest_xlast = np.mean(probs)

                    sys.stderr.write("P(y > y_best) = {}\n".format(
                        prob_gt_ybest_xlast))

                    res = self.predict(fits, thin=thin)
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
                            sys.stderr.write("std_predictive_threshold is set. Checking the predictive_std first\n")
                            sys.stderr.write("predictive_std: {}\n".format(
                                predictive_std))
                            if predictive_std < self.predictive_std_threshold:
                                sys.stderr.write("Predicting..\n")
                                result = {'predictive_mean': predictive_mean,
                                          'predictive_std': predictive_std,
                                          'found': found,
                                          'prob_gt_ybest_xlast':
                                              prob_gt_ybest_xlast,
                                          'terminate': found}
                            else:
                                sys.stderr.write("Continue Training..\n")
                                result = {'predictive_mean': predictive_mean,
                                          'predictive_std': predictive_std,
                                          'found': found,
                                          'prob_gt_ybest_xlast':
                                              prob_gt_ybest_xlast,
                                          'terminate': False}
                    else:
                        sys.stderr.write("Continue Training..\n")
                        result = {'predictive_mean': predictive_mean,
                                  'predictive_std': predictive_std,
                                  'found': found,
                                  'prob_gt_ybest_xlast': prob_gt_ybest_xlast,
                                  'terminate': False}

        pprint(result)
        return result


class NewConservativeTerminationCriterionV3(NewTerminationCriterion):
    """New Conservative Termination Criterion.

    This Termination Criterion is a bit less conservative. Along with the
    standard conditions, it also looks at the distance between the
    mean + N standard deviations of the prediction to see if that number is
    within the threshold. If not, terminate. Also discard runs for which
    offsets cause the run to pass the threshold.

    Extends:
        NewTerminationCriterion

    Variables:
        N: margin of standard deviation to consider when terminating
    """

    n = None
    predictive_std_threshold = None

    def __init__(self, y_lists, curr_y_list, y_best,
                 xlim, nthreads, prob_x_greater_type,
                 predictive_std_threshold=None, n=5):
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
        self.n = n
        super(NewConservativeTerminationCriterionV3,
              self).__init__(y_lists, curr_y_list, y_best,
                             xlim, nthreads, prob_x_greater_type)

    def run(self, thin=PREDICTION_THINNING,
            threshold=IMPROVEMENT_PROB_THRESHOLD):
        """Run method for the ConservativeTerminationCriterion class.

        The actual implementation of the termination criterion.

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
        sys.stderr.write("Current List: {}\n".format(self.curr_y_list))
        if len(self.curr_y_list) == 0:
            sys.stderr.write("No y_s done yet\n")
            result = {"predictive_mean": None,
                      "predictive_std": None,
                      "found": False,
                      'prob_gt_ybest_xlast': 0,
                      "terminate": False}
        else:
            y_curr_best = np.max(self.curr_y_list)
            if y_curr_best > self.y_best:
                sys.stderr.write("Already exceeding best. Proceed\n")
                result = {"predictive_mean": y_curr_best,
                          "predictive_std": 0,
                          "found": True,
                          'prob_gt_ybest_xlast': 1.0,
                          "terminate": False}
            else:
                fits = []
                for idx in xrange(len(self.models)):
                    y = self.y_lists[idx]
                    # cut_beginning(y_list)
                    x = np.asarray(range(1, len(y) + 1))
                    result = None
                    fits.append(self.models[idx].fit(x, y))

                if not np.any(fits):
                    sys.stderr.write('Failed in fitting, Let the training proceed in any case\n')
                    result = {'predictive_mean': y_curr_best,
                              'predictive_std': 0,
                              'found': False,
                              'prob_gt_ybest_xlast': 0,
                              'terminate': False}
                else:
                    probs = []
                    if self.prob_x_greater_type == \
                            'posterior_prob_x_greater_than':
                        for i in range(len(self.models)):
                            if fits[i] is True:
                                model = self.models[i]
                                probs.append(
                                    model.posterior_prob_x_greater_than(
                                        self.xlim,
                                        self.y_best,
                                        thin=thin))
                    else:
                        for i in range(len(self.models)):
                            if fits[i] is True:
                                model = self.models[i]
                                probs.append(
                                    model.posterior_mean_prob_x_greater_than(
                                        self.xlim,
                                        self.y_best,
                                        thin=thin))

                    # Not sure if averaging probabilities makes sense:
                    prob_gt_ybest_xlast = np.mean(probs)

                    sys.stderr.write("P(y > y_best) = {}\n".format(
                        prob_gt_ybest_xlast))

                    res = self.predict(fits, thin=thin)
                    predictive_mean = res['predictive_mean']
                    predictive_std = res['predictive_std']
                    found = res['found']
                    if found is False:
                        result = {'predictive_mean': predictive_mean,
                                  'predictive_std': predictive_std,
                                  'found': False,
                                  'prob_gt_ybest_xlast': prob_gt_ybest_xlast,
                                  'terminate': False}
                    else:
                        # Condition if predictive mean + n*predictive_std is
                        # less than best result, terminate
                        upper_bound = predictive_mean + self.n * predictive_std
                        if upper_bound < self.y_best:
                            sys.stderr.write('The predictive mean {} is more than {} standard deviations away from the y_best {}. Terminate.\n'.format(
                                predictive_mean, self.n, self.y_best))
                            result = {'predictive_mean': predictive_mean,
                                      'predictive_std': predictive_std,
                                      'found': True,
                                      'prob_gt_ybest_xlast': 0.0,
                                      'terminate': True}
                        else:
                            if prob_gt_ybest_xlast < threshold:
                                # Below the threshold. Send the termination
                                # signals
                                if self.predictive_std_threshold is None:
                                    result = {
                                        'predictive_mean': predictive_mean,
                                        'predictive_std': predictive_std,
                                        'found': found,
                                        'prob_gt_ybest_xlast':
                                        prob_gt_ybest_xlast,
                                        'terminate': found}
                                else:
                                    sys.stderr.write(
                                        "std_predictive_threshold is set. Checking the predictive_std first\n")
                                    sys.stderr.write(
                                        "predictive_std: {}\n".format(
                                            predictive_std))
                                    if (predictive_std <
                                            self.predictive_std_threshold):
                                        sys.stderr.write("Predicting...\n")
                                        result = {
                                            'predictive_mean': predictive_mean,
                                            'predictive_std': predictive_std,
                                            'found': found,
                                            'prob_gt_ybest_xlast':
                                            prob_gt_ybest_xlast,
                                            'terminate': found}
                                    else:
                                        sys.stderr.write("Continue Training..\n")
                                        result = {
                                            'predictive_mean': predictive_mean,
                                            'predictive_std': predictive_std,
                                            'found': found,
                                            'prob_gt_ybest_xlast':
                                            prob_gt_ybest_xlast,
                                            'terminate': False}
                            else:
                                sys.stderr.write("Continue Training..\n")
                                result = {
                                    'predictive_mean': predictive_mean,
                                    'predictive_std': predictive_std,
                                    'found': found,
                                    'prob_gt_ybest_xlast': prob_gt_ybest_xlast,
                                    'terminate': False}

        pprint(result)
        return result


class NewConservativeTerminationCriterionV4(NewTerminationCriterionV2):
    """New Conservative Termination Criterion.

    This Termination Criterion is a bit less conservative. Along with the
    standard conditions, it also looks at the distance between the
    mean + N standard deviations of the prediction to see if that number is
    within the threshold. If not, terminate. Also discard runs for which
    offsets cause the run to pass the threshold. Also, discard runs in
    prev_lists which do not have more than half of validations done.

    Extends:
        NewTerminationCriterion

    Variables:
        N: margin of standard deviation to consider when terminating
    """

    n = None
    predictive_std_threshold = None

    def __init__(self, y_lists, curr_y_list, y_best,
                 xlim, nthreads, prob_x_greater_type,
                 predictive_std_threshold=None, n=5):
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
        self.n = n
        super(NewConservativeTerminationCriterionV4,
              self).__init__(y_lists, curr_y_list, y_best,
                             xlim, nthreads, prob_x_greater_type)

    def run(self, thin=PREDICTION_THINNING,
            threshold=IMPROVEMENT_PROB_THRESHOLD):
        """Run method for the ConservativeTerminationCriterion class.

        The actual implementation of the termination criterion.

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
        sys.stderr.write("Current List: {}\n".format(self.curr_y_list))
        if len(self.curr_y_list) == 0:
            sys.stderr.write("No y_s done yet\n")
            result = {"predictive_mean": None,
                      "predictive_std": None,
                      "found": False,
                      'prob_gt_ybest_xlast': 0,
                      "terminate": False}
        else:
            y_curr_best = np.max(self.curr_y_list)
            if y_curr_best > self.y_best:
                sys.stderr.write("Already exceeding best. Proceed\n")
                result = {"predictive_mean": y_curr_best,
                          "predictive_std": 0,
                          "found": True,
                          'prob_gt_ybest_xlast': 1.0,
                          "terminate": False}
            else:
                fits = []
                for idx in xrange(len(self.models)):
                    y = self.y_lists[idx]
                    # cut_beginning(y_list)
                    x = np.asarray(range(1, len(y) + 1))
                    result = None
                    fits.append(self.models[idx].fit(x, y))

                if not np.any(fits):
                    sys.stderr.write('Failed in fitting, Let the training proceed in any case\n')
                    result = {'predictive_mean': y_curr_best,
                              'predictive_std': 0,
                              'found': False,
                              'prob_gt_ybest_xlast': 0,
                              'terminate': False}
                else:
                    probs = []
                    if self.prob_x_greater_type == \
                            'posterior_prob_x_greater_than':
                        for i in range(len(self.models)):
                            if fits[i] is True:
                                model = self.models[i]
                                probs.append(
                                    model.posterior_prob_x_greater_than(
                                        self.xlim,
                                        self.y_best,
                                        thin=thin))
                    else:
                        for i in range(len(self.models)):
                            if fits[i] is True:
                                model = self.models[i]
                                probs.append(
                                    model.posterior_mean_prob_x_greater_than(
                                        self.xlim,
                                        self.y_best,
                                        thin=thin))

                    # Not sure if averaging probabilities makes sense:
                    prob_gt_ybest_xlast = np.mean(probs)

                    sys.stderr.write("P(y > y_best) = {}\n".format(
                        prob_gt_ybest_xlast))

                    res = self.predict(fits, thin=thin)
                    predictive_mean = res['predictive_mean']
                    predictive_std = res['predictive_std']
                    found = res['found']
                    if found is False:
                        result = {'predictive_mean': predictive_mean,
                                  'predictive_std': predictive_std,
                                  'found': False,
                                  'prob_gt_ybest_xlast': prob_gt_ybest_xlast,
                                  'terminate': False}
                    else:
                        # Condition if predictive mean + n*predictive_std is
                        # less than best result, terminate
                        upper_bound = predictive_mean + self.n * predictive_std
                        if upper_bound < self.y_best:
                            sys.stderr.write('The predictive mean {} is more than {} standard deviations away from the y_best {}. Terminate.\n'.format(
                                predictive_mean, self.n, self.y_best))
                            result = {'predictive_mean': predictive_mean,
                                      'predictive_std': predictive_std,
                                      'found': True,
                                      'prob_gt_ybest_xlast': 0.0,
                                      'terminate': True}
                        else:
                            if prob_gt_ybest_xlast < threshold:
                                # Below the threshold. Send the termination
                                # signals
                                if self.predictive_std_threshold is None:
                                    result = {
                                        'predictive_mean': predictive_mean,
                                        'predictive_std': predictive_std,
                                        'found': found,
                                        'prob_gt_ybest_xlast':
                                        prob_gt_ybest_xlast,
                                        'terminate': found}
                                else:
                                    sys.stderr.write(
                                        "std_predictive_threshold is set. Checking the predictive_std first\n")
                                    sys.stderr.write(
                                        "predictive_std: {}\n".format(
                                            predictive_std))
                                    if (predictive_std <
                                            self.predictive_std_threshold):
                                        sys.stderr.write("Predicting...\n")
                                        result = {
                                            'predictive_mean': predictive_mean,
                                            'predictive_std': predictive_std,
                                            'found': found,
                                            'prob_gt_ybest_xlast':
                                            prob_gt_ybest_xlast,
                                            'terminate': found}
                                    else:
                                        sys.stderr.write("Continue Training..\n")
                                        result = {
                                            'predictive_mean': predictive_mean,
                                            'predictive_std': predictive_std,
                                            'found': found,
                                            'prob_gt_ybest_xlast':
                                            prob_gt_ybest_xlast,
                                            'terminate': False}
                            else:
                                sys.stderr.write("Continue Training..\n")
                                result = {
                                    'predictive_mean': predictive_mean,
                                    'predictive_std': predictive_std,
                                    'found': found,
                                    'prob_gt_ybest_xlast': prob_gt_ybest_xlast,
                                    'terminate': False}

        pprint(result)
        return result
