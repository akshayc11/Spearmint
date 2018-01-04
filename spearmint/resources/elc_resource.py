"""Methods and Classes for resources for Extrapolation of learning curves.

curves. Modelled on the resources.py structure.
"""

import importlib
import logging
import sys

from operator import add


def parse_elc_config(elc_resource_config=None):
    """Get fully defined ELC config dict.

    Parses provided config related to the extrapolation of learning curves
    and returns a fully filled one.

    Arguments:
        elc_resource_config {dict} -- dictionary of parameters for ELC resource

    Returns:
        dict -- fully specified dictionary of ELC parameters
    """
    if elc_resource_config is None:
        elc_resource_config = {}

    if 'name' not in elc_resource_config:
        elc_resource_config['name'] = 'Elc'
    
    if 'scheduler' not in elc_resource_config:
        elc_resource_config['scheduler'] = 'local'
    
    if 'elc_task_name' not in elc_resource_config:
        elc_resource_config['elc_task_name'] = 'Elc_Task'
    
    if 'nthreads' not in elc_resource_config:
        elc_resource_config['nthreads'] = 4
    else:
        assert isinstance(elc_resource_config['nthreads'], int)
    
    if 'mode' not in elc_resource_config:
        elc_resource_config['mode'] = 'conservative'
    else:
        assert elc_resource_config['mode'] == 'conservative' or \
            elc_resource_config['mode'] == 'optimistic' or \
            elc_resource_config['mode'] == 'gelc-conservative' or \
            elc_resource_config['mode'] == 'gelc-hp-conservative', 'mode should be \
            conservative or optimistic or gelc-conservative or gelc-hp-conservative'

    if 'prob_x_greater_type' not in elc_resource_config:
        elc_resource_config['prob_x_greater_type'] = \
            'posterior_mean_prob_x_greater_than'
    else:
        assert elc_resource_config['prob_x_greater_type'] == \
            'posterior_mean_prob_x_greater_than' or \
            elc_resource_config['prob_x_greater_type'] == \
            'posterior_prob_x_greater_than', 'prob_x_greater_type \
            should be posterior_mean_prob_x_greater_than or \
            posterior_prob_x_greater_than'
    
    if 'threshold' not in elc_resource_config:
        elc_resource_config['threshold'] = 0.05
    else:
        assert elc_resource_config['threshold'] > .0 and \
            elc_resource_config['threshold'] < 1.0, 'threshold must be \
            between 0 and 1'
    
    if 'predictive_std_threshold' not in elc_resource_config:
        elc_resource_config['predictive_std_threshold'] = 0.005
    else:
        assert elc_resource_config['predictive_std_threshold'] > 0.0 or \
            elc_resource_config['predictive_std_threshold'] is None, '\
            predictive_std_threshold should be greater than 0.0 or specified \
            as None'
    
    if 'scheduler' not in elc_resource_config:
        elc_resource_config['scheduler'] = 'local'
    
    if 'xlim' not in elc_resource_config:
        raise Exception('xlim is a required parameter for the resource_config')
    else:
        assert elc_resource_config['xlim'] > 0

    if 'min_y_prev' not in elc_resource_config:
        elc_resource_config['min_y_prev'] = 1

    if 'recency-weighting' not in elc_resource_config:
        elc_resource_config['recency-weighting'] = False

    if 'monotonicity-condition' not in elc_resource_config:
        elc_resource_config['monotonicity-condition'] = False

    if 'selection' not in elc_resource_config:
        elc_resource_config['selection'] = 'covariance'

    
    return elc_resource_config


def parse_elc_resource_from_config(config):
    """Return a dict of resources for the extrapolation of learning curves.

    Based on the config specifications for extrapolation of learning curves,
    this function returns a dictionary of resources to be used for the
    extrapolation of learning curves.

    Arguments:
        config {dict} -- original config dict

    Returns:
        {dict} -- dictionary of resources for extrapolation of learning curves
    """
    if "elc_resource" not in config:
        default_elc_resource_name = 'Elc'
        task_names = ['Elc_Task']
        elc_resource_config = parse_elc_config()
        elc_resource = elc_resource_factory(
            default_elc_resource_name,
            task_names,
            elc_resource_config)
    else:
        elc_resource_config = config['elc_resource']
        elc_resource_config = parse_elc_config(elc_resource_config)
        elc_resource_name = elc_resource_config['name']
        task_names = [elc_resource_config.get('elc_task_name', 'Elc_Task')]
        elc_resource = elc_resource_factory(
            elc_resource_name,
            task_names,
            elc_resource_config)

    return elc_resource


def elc_resource_factory(resource_name, task_names, resource_config):
    """Factory for generating ELC Resource object.

    Generates the ELC Resource object based on the specifications.

    Arguments:
        resource_name {str} -- name of resource.
        task_names {[str]} -- list of task names to be handled by resource
        resource_config {dict} -- config options for resource

    Returns:
        [type] -- [description]
    """
    scheduler_class = resource_config['scheduler']
    scheduler_object = importlib.import_module('spearmint.schedulers.' +
                                               scheduler_class).init(
                                                   resource_config)
    max_concurrent = 1

    resource = ElcResource(resource_name, task_names, scheduler_object,
                           scheduler_class, max_concurrent, resource_config)
    return resource


class ElcResource(object):
    """ElcResource object class.

    The ElcResource object class contains methods to submit jobs for performing
    extrapolation of learning curves.

    Variables:
        name {str} -- name of resource
        scheduler {AbstractScheduler} -- scheduler object
        scheduler_class {str} -- type of scheduler object (should inherit
                                 AbstractScheduler)
        max_concurrent {int} -- maximum number of concurrent jobs that can run
        resource_config {dict} -- dictionary with terminationcriterion
                                  information
        tasks {list} -- list of tasks this object caters to.
    """

    name = None
    scheduler = None
    scheduler_class = None
    max_concurrent = None
    max_finished_jobs = None
    resource_config = None
    tasks = None

    def __init__(self, name, tasks, scheduler, scheduler_class, max_concurrent,
                 resource_config):
        """Constructor for ELC resource object.

        Constructor for ELC resource object.

        Arguments:
            name {str} -- name of resource
            tasks {list} -- list of tasks this object caters to.
                name {str} -- [description]
            scheduler {AbstractScheduler} -- scheduler object
            scheduler_class {str} -- type of scheduler object (should inherit
                                     AbstractScheduler)
            max_concurrent {int} -- maximum number of concurrent jobs that can
                                    run
            max_finished_jobs {int} -- maximum number of Elc checks
            resource_config {dict} -- dictionary with terminationcriterion
                                      information
        """
        self.name = name
        self.scheduler = scheduler
        self.scheduler_class = scheduler_class
        self.max_concurrent = max_concurrent
        self.tasks = tasks
        self.resource_config = resource_config

        if len(tasks) == 0:
            logging.warn("Resource {} has not assigned tasks".format(
                self.name))

    def num_pending(self, jobs):
        """Number of pending jobs.

        Number of pending jobs.

        Arguments:
            jobs {[list]} -- list of jobs

        Returns:
            [int] -- number of pending elc jobs
        """
        pending_jobs = map(lambda x: x['elc_status'] in ['pending', 'new'],
                           jobs)
        return reduce(add, pending_jobs, 0)

    def accepting_jobs(self, jobs):
        """If the current resource can accept more jobs.

        Says if the current resource can accept more jobs.

        Arguments:
            jobs {[list]} -- list of jobs

        Returns:
            bool -- whether more elc jobs can start
        """
        if self.num_pending(jobs) >= self.max_concurrent:
            return False
        else:
            return True

    def is_elc_alive(self, job):
        """Check if given elc job alive.

        is given elc job alive?

        Arguments:
            job {dict} -- dict of job

        Returns:
            bool -- whether job is alive.
        """
        return self.scheduler.alive(job['elc_proc_id'])

    def kill_elc(self, job):
        if self.is_elc_alive(job):
            self.scheduler.kill(job['elc_process'])

    def attempt_dispatch(self, experiment_name, job, db_address, expt_dir):
        """Attempt to dispatch an ELC job using the elc scheduler.

        [description]

        Arguments:
            experiment_name {[type]} -- [description]
            job {[type]} -- [description]
            db_address {[type]} -- [description]
            expt_dir {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        mode = self.resource_config['mode']
        prob_x_greater_type = self.resource_config['prob_x_greater_type']
        nthreads = self.resource_config['nthreads']
        threshold = self.resource_config['threshold']
        predictive_std_threshold = self.resource_config[
            'predictive_std_threshold']
        xlim = self.resource_config['xlim']
        min_y_prev = self.resource_config['min_y_prev']
        recency_weighting = self.resource_config['recency-weighting']
        monotonicity_condition = self.resource_config['monotonicity-condition']
        process = self.scheduler.submit_elc(job['id'], experiment_name,
                                               expt_dir, db_address,
                                               mode,
                                               prob_x_greater_type,
                                               threshold,
                                               predictive_std_threshold,
                                               nthreads,
                                               xlim,
                                               min_y_prev=min_y_prev,
                                               recency_weighting=recency_weighting,
                                               monotonicity_condition=monotonicity_condition)
        process_id = process.pid
        if process_id is not None:
            sys.stderr.write('Submitted elc job {} with {} scheduler. \
                             (process_id: {})\n'.format(job['id'],
                                                        self.scheduler_class,
                                                        process_id))
        else:
            sys.stderr.write('Failed to submit elc job {}.\n'.format(
                job['id']))
        return process
