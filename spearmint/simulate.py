import importlib
import optparse
import os
import sys
import time

from collections import OrderedDict

import numpy as np

np.set_printoptions(precision=5)

try:
    import simplejson as json
except:
    import json
try:
    import cPickle as pickle
except:
    import pickle

from bson.binary import Binary

from main import *
from spearmint.utils.cleanup import cleanup

class SimClass(object):
    experiment_dir = None
    config_file = None

    resources = None
    chooser = None
    experiment_name = None
    db_address = None
    db = None
    
    
    def __init__(self, expt_dir, config_file, expt_name):
        self.experiment_dir = expt_dir
        self.config_file = config_file
        self.options = get_options_from_config(self.config_file)
        self.options['experiment-name'] = expt_name
        cleanup(self.experiment_dir, config_file=os.path.basename(config_file), expt_name=expt_name)
        self.resources, self.chooser, self.experiment_name, self.db_address, self.db = prepare(self.options)
        
        resource_name = self.resources.keys()[0]
        resource = self.resources[resource_name]
        task_names = resource.tasks
        task_options = {task:self.options['tasks'][task] for task in task_names}
        task_group = load_task_group(self.db, self.options, task_names)
        hypers = load_hypers(self.db, self.experiment_name)
        hypers = self.chooser.fit(task_group, hypers, task_options)


    def iteration(self, **kwargs):
        resource_name = self.resources.keys()[0]
        resource = self.resources[resource_name]
        
        task_names = resource.tasks
        task_options = {task:self.options['tasks'][task] for task in task_names}
        task_group = load_task_group(self.db, self.options, task_names)
        jobs = load_jobs(self.db, self.experiment_name)
        job_id = len(jobs) + 1
        suggested_input = self.make_suggested_input(**kwargs)
        if kwargs['values'] != None:
            values = {self.chooser.objective['name']: kwargs['values']}
        else:
            values = None
        job = {
            'id': job_id,
            'params': task_group.paramify(suggested_input),
            'expt_dir': self.experiment_dir,
            'tasks': task_names,
            'resource': resource_name,
            'main-file': None,
            'language': None,
            'status': 'complete',
            'submit time': time.time(),
            'start time': time.time(),
            'end time': time.time(),
            'validation_accs': None,
            'validation_updated': False,
            'validation check time': -1,
            'y_best': None,
            'elc_result': None,
            'elc_status': None,
            'elc_proc_id': None,
            'process': None,
            'elc_process': None,
            'cov_ids': None,
            'cov_list': None,
            'values': values,
        }
        save_job(job, self.db, self.experiment_name)
        if values is not None:
            hypers = load_hypers(self.db, self.experiment_name)
            hypers = self.chooser.fit(task_group, hypers, task_options)
            save_hypers(hypers, self.db, self.experiment_name)
        
        return job

    def get_cov(self):
        jobs = load_jobs(self.db, self.experiment_name)
        curr_input = [jobs[-1]['params']]
        prev_inputs = [j['params'] for j in jobs[:-1]]
        covs = self.chooser.get_cov(curr_input, prev_inputs, debug=False)
        covs = [c for c in covs]
        return covs

    def make_suggested_input(self, **kwargs):
        # Make the suggested input in the format as needed
        resource_name = self.resources.keys()[0]
        resource = self.resources[resource_name]
        task_names = resource.tasks
        task_group = load_task_group(self.db, self.options, task_names)
        dummy_task = task_group.dummy_task
        var_meta = dummy_task.variables_meta
        params = []
        for name, vdict in var_meta.iteritems():
            params.append(kwargs[name])
        return np.array(params)
