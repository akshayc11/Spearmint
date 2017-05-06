# -*- coding: utf-8 -*-
# Spearmint
#
# Academic and Non-Commercial Research Use Software License and Terms
# of Use
#
# Spearmint is a software package to perform Bayesian optimization
# according to specific algorithms (the “Software”).  The Software is
# designed to automatically run experiments (thus the code name
# 'spearmint') in a manner that iteratively adjusts a number of
# parameters so as to minimize some objective in as few runs as
# possible.
#
# The Software was developed by Ryan P. Adams, Michael Gelbart, and
# Jasper Snoek at Harvard University, Kevin Swersky at the
# University of Toronto (“Toronto”), and Hugo Larochelle at the
# Université de Sherbrooke (“Sherbrooke”), which assigned its rights
# in the Software to Socpra Sciences et Génie
# S.E.C. (“Socpra”). Pursuant to an inter-institutional agreement
# between the parties, it is distributed for free academic and
# non-commercial research use by the President and Fellows of Harvard
# College (“Harvard”).
#
# Using the Software indicates your agreement to be bound by the terms
# of this Software Use Agreement (“Agreement”). Absent your agreement
# to the terms below, you (the “End User”) have no rights to hold or
# use the Software whatsoever.
#
# Harvard agrees to grant hereunder the limited non-exclusive license
# to End User for the use of the Software in the performance of End
# User’s internal, non-commercial research and academic use at End
# User’s academic or not-for-profit research institution
# (“Institution”) on the following terms and conditions:
#
# 1.  NO REDISTRIBUTION. The Software remains the property Harvard,
# Toronto and Socpra, and except as set forth in Section 4, End User
# shall not publish, distribute, or otherwise transfer or make
# available the Software to any other party.
#
# 2.  NO COMMERCIAL USE. End User shall not use the Software for
# commercial purposes and any such use of the Software is expressly
# prohibited. This includes, but is not limited to, use of the
# Software in fee-for-service arrangements, core facilities or
# laboratories or to provide research services to (or in collaboration
# with) third parties for a fee, and in industry-sponsored
# collaborative research projects where any commercial rights are
# granted to the sponsor. If End User wishes to use the Software for
# commercial purposes or for any other restricted purpose, End User
# must execute a separate license agreement with Harvard.
#
# Requests for use of the Software for commercial purposes, please
# contact:
#
# Office of Technology Development
# Harvard University
# Smith Campus Center, Suite 727E
# 1350 Massachusetts Avenue
# Cambridge, MA 02138 USA
# Telephone: (617) 495-3067
# Facsimile: (617) 495-9568
# E-mail: otd@harvard.edu
#
# 3.  OWNERSHIP AND COPYRIGHT NOTICE. Harvard, Toronto and Socpra own
# all intellectual property in the Software. End User shall gain no
# ownership to the Software. End User shall not remove or delete and
# shall retain in the Software, in any modifications to Software and
# in any Derivative Works, the copyright, trademark, or other notices
# pertaining to Software as provided with the Software.
#
# 4.  DERIVATIVE WORKS. End User may create and use Derivative Works,
# as such term is defined under U.S. copyright laws, provided that any
# such Derivative Works shall be restricted to non-commercial,
# internal research and academic use at End User’s Institution. End
# User may distribute Derivative Works to other Institutions solely
# for the performance of non-commercial, internal research and
# academic use on terms substantially similar to this License and
# Terms of Use.
#
# 5.  FEEDBACK. In order to improve the Software, comments from End
# Users may be useful. End User agrees to provide Harvard with
# feedback on the End User’s use of the Software (e.g., any bugs in
# the Software, the user experience, etc.).  Harvard is permitted to
# use such information provided by End User in making changes and
# improvements to the Software without compensation or an accounting
# to End User.
#
# 6.  NON ASSERT. End User acknowledges that Harvard, Toronto and/or
# Sherbrooke or Socpra may develop modifications to the Software that
# may be based on the feedback provided by End User under Section 5
# above. Harvard, Toronto and Sherbrooke/Socpra shall not be
# restricted in any way by End User regarding their use of such
# information.  End User acknowledges the right of Harvard, Toronto
# and Sherbrooke/Socpra to prepare, publish, display, reproduce,
# transmit and or use modifications to the Software that may be
# substantially similar or functionally equivalent to End User’s
# modifications and/or improvements if any.  In the event that End
# User obtains patent protection for any modification or improvement
# to Software, End User agrees not to allege or enjoin infringement of
# End User’s patent against Harvard, Toronto or Sherbrooke or Socpra,
# or any of the researchers, medical or research staff, officers,
# directors and employees of those institutions.
#
# 7.  PUBLICATION & ATTRIBUTION. End User has the right to publish,
# present, or share results from the use of the Software.  In
# accordance with customary academic practice, End User will
# acknowledge Harvard, Toronto and Sherbrooke/Socpra as the providers
# of the Software and may cite the relevant reference(s) from the
# following list of publications:
#
# Practical Bayesian Optimization of Machine Learning Algorithms
# Jasper Snoek, Hugo Larochelle and Ryan Prescott Adams
# Neural Information Processing Systems, 2012
#
# Multi-Task Bayesian Optimization
# Kevin Swersky, Jasper Snoek and Ryan Prescott Adams
# Advances in Neural Information Processing Systems, 2013
#
# Input Warping for Bayesian Optimization of Non-stationary Functions
# Jasper Snoek, Kevin Swersky, Richard Zemel and Ryan Prescott Adams
# Preprint, arXiv:1402.0929, http://arxiv.org/abs/1402.0929, 2013
#
# Bayesian Optimization and Semiparametric Models with Applications to
# Assistive Technology Jasper Snoek, PhD Thesis, University of
# Toronto, 2013
#
# 8.  NO WARRANTIES. THE SOFTWARE IS PROVIDED "AS IS." TO THE FULLEST
# EXTENT PERMITTED BY LAW, HARVARD, TORONTO AND SHERBROOKE AND SOCPRA
# HEREBY DISCLAIM ALL WARRANTIES OF ANY KIND (EXPRESS, IMPLIED OR
# OTHERWISE) REGARDING THE SOFTWARE, INCLUDING BUT NOT LIMITED TO ANY
# IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, OWNERSHIP, AND NON-INFRINGEMENT.  HARVARD, TORONTO AND
# SHERBROOKE AND SOCPRA MAKE NO WARRANTY ABOUT THE ACCURACY,
# RELIABILITY, COMPLETENESS, TIMELINESS, SUFFICIENCY OR QUALITY OF THE
# SOFTWARE.  HARVARD, TORONTO AND SHERBROOKE AND SOCPRA DO NOT WARRANT
# THAT THE SOFTWARE WILL OPERATE WITHOUT ERROR OR INTERRUPTION.
#
# 9.  LIMITATIONS OF LIABILITY AND REMEDIES. USE OF THE SOFTWARE IS AT
# END USER’S OWN RISK. IF END USER IS DISSATISFIED WITH THE SOFTWARE,
# ITS EXCLUSIVE REMEDY IS TO STOP USING IT.  IN NO EVENT SHALL
# HARVARD, TORONTO OR SHERBROOKE OR SOCPRA BE LIABLE TO END USER OR
# ITS INSTITUTION, IN CONTRACT, TORT OR OTHERWISE, FOR ANY DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, CONSEQUENTIAL, PUNITIVE OR OTHER
# DAMAGES OF ANY KIND WHATSOEVER ARISING OUT OF OR IN CONNECTION WITH
# THE SOFTWARE, EVEN IF HARVARD, TORONTO OR SHERBROOKE OR SOCPRA IS
# NEGLIGENT OR OTHERWISE AT FAULT, AND REGARDLESS OF WHETHER HARVARD,
# TORONTO OR SHERBROOKE OR SOCPRA IS ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGES.
#
# 10. INDEMNIFICATION. To the extent permitted by law, End User shall
# indemnify, defend and hold harmless Harvard, Toronto and Sherbrooke
# and Socpra, their corporate affiliates, current or future directors,
# trustees, officers, faculty, medical and professional staff,
# employees, students and agents and their respective successors,
# heirs and assigns (the "Indemnitees"), against any liability,
# damage, loss or expense (including reasonable attorney's fees and
# expenses of litigation) incurred by or imposed upon the Indemnitees
# or any one of them in connection with any claims, suits, actions,
# demands or judgments arising from End User’s breach of this
# Agreement or its Institution’s use of the Software except to the
# extent caused by the gross negligence or willful misconduct of
# Harvard, Toronto or Sherbrooke or Socpra. This indemnification
# provision shall survive expiration or termination of this Agreement.
#
# 11. GOVERNING LAW. This Agreement shall be construed and governed by
# the laws of the Commonwealth of Massachusetts regardless of
# otherwise applicable choice of law standards.
#
# 12. NON-USE OF NAME.  Nothing in this License and Terms of Use shall
# be construed as granting End Users or their Institutions any rights
# or licenses to use any trademarks, service marks or logos associated
# with the Software.  You may not use the terms “Harvard” or
# “University of Toronto” or “Université de Sherbrooke” or “Socpra
# Sciences et Génie S.E.C.” (or a substantially similar term) in any
# way that is inconsistent with the permitted uses described
# herein. You agree not to use any name or emblem of Harvard, Toronto
# or Sherbrooke, or any of their subdivisions for any purpose, or to
# falsely suggest any relationship between End User (or its
# Institution) and Harvard, Toronto and/or Sherbrooke, or in any
# manner that would infringe or violate any of their rights.
#
# 13. End User represents and warrants that it has the legal authority
# to enter into this License and Terms of Use on behalf of itself and
# its Institution.

import argparse
import os
import sys
import time

import numpy as np

from spearmint.utils.database.mongodb import MongoDB
from spearmint.pylrpredictor.terminationcriterion import \
    ConservativeTerminationCriterion, OptimisticTerminationCriterion
from spearmint.pylrpredictor.prevonlyterminationcriterion import ConservativeTerminationCriterion as GelcConservativeTerminationCriterion
from spearmint.pylrpredictor.prevonlyhpterminationcriterion import ConservativeTerminationCriterion as GelcHpConservativeTerminationCriterion

def main():
    parser = argparse.ArgumentParser(description="usage: %prog [options]")
    parser.add_argument('--experiment-name',
                        help='The name of the experiment in the database',
                        type=str,
                        required=True)
    parser.add_argument('--database-address', dest="db_address",
                        help='The address where the database is located',
                        type=str,
                        required=True)
    parser.add_argument('--job-id',
                        help='The id number of the job to launch \
                            in the database',
                        type=int,
                        required=True)
    parser.add_argument('--validation',
                        help='Specify true to get validation results for a \
                            job',
                        type=bool,
                        default=False)
    parser.add_argument('--elc',
                        help='Specify true to submit ELC job',
                        type=bool,
                        default=False)
    parser.add_argument('--mode',
                        help='Specify mode for termination criterion',
                        type=str,
                        default='conservative')
    parser.add_argument('--prob-x-greater-type',
                        help='type of probability sampling',
                        type=str,
                        default='posterior_prob_mean_x_greater_than')
    parser.add_argument('--threshold',
                        help='probability threshold for ELC',
                        type=float,
                        default=0.05)
    parser.add_argument('--predictive-std-threshold',
                        help='threshold for std dev of prediction',
                        type=float,
                        default=0.005)
    parser.add_argument('--nthreads',
                        help='number of threads for MCMC',
                        type=int,
                        default=4)
    parser.add_argument('--xlim',
                        help='total number of validation runs permitted',
                        type=int,
                        default=None)
    parser.add_argument('--min-y-prev',
                        help='number of previous builds required for gelc',
                        type=int,
                        default=1)
    parser.add_argument('--recency-weighting',
                        type=bool,
                        default=False,
                        help='Recency weighting during loss computation?')
    parser.add_argument('--monotonicity-condition',
                        type=bool,
                        default=False,
                        help='Flag to enforce that final value should not be lesser than best seen so far')
    options = parser.parse_args()

    launch(options)


def launch(options):
    """
    Launches a job from on a given id.
    """
    db_address = options.db_address
    experiment_name = options.experiment_name
    job_id = options.job_id
    validation = options.validation
    elc = options.elc
    mode = options.mode
    prob_x_greater_type = options.prob_x_greater_type
    threshold = options.threshold
    predictive_std_threshold = options.predictive_std_threshold
    nthreads = options.nthreads
    xlim = options.xlim
    min_y_prev = options.min_y_prev
    recency_weighting = options.recency_weighting
    monotonicity_condition = options.monotonicity_condition

    db = MongoDB(database_address=db_address)
    job = db.load(experiment_name, 'jobs', {'id': job_id})
    prev_jobs = db.load(experiment_name, 'jobs', {'status': 'complete'})
    if validation is False and elc is False:
        # We want to launch this job.
        start_time = time.time()
        job = db.load(experiment_name, 'jobs', {'id': job_id})
        job['start time'] = start_time
        db.save(job, experiment_name, 'jobs', {'id': job_id})

        sys.stderr.write("Job launching after %0.2f seconds in submission.\n"
                         % (start_time - job['submit time']))
        success = False

        try:
            if job['language'].lower() == 'python':
                result = python_launcher(job)

            # elif job['language'].lower() == 'matlab':
            #     result = matlab_launcher(job)

            # elif job['language'].lower() == 'shell':
            #     result = shell_launcher(job)

            # elif job['language'].lower() == 'mcr':
            #     result = mcr_launcher(job)

            else:
                raise Exception("That language has not been implemented.")

            if not isinstance(result, dict):
                # Returning just NaN means NaN on all tasks
                if np.isnan(result):
                    # Apparently this dict generator throws an error for
                    # some people??
                    # result = {task_name: np.nan
                    #           for task_name in job['tasks']}
                    # So we use the much uglier version below... ????
                    result = dict(zip(job['tasks'],
                                      [np.nan] * len(job['tasks'])))
                elif len(job['tasks']) == 1:  # Only one named job
                    result = {job['tasks'][0]: result}
                else:
                    result = {'main': result}

            if set(result.keys()) != set(job['tasks']):
                raise Exception("Result task names %s did not match job \
                    task names %s." % (result.keys(), job['tasks']))
            
            success = True
        except:
            import traceback
            traceback.print_exc()
            sys.stderr.write("Problem executing the function\n")
            print sys.exc_info()

        end_time = time.time()

        if success:
            sys.stderr.write("Completed successfully in %0.2f seconds. [%s]\n"
                             % (end_time - start_time, result))
            job = db.load(experiment_name, 'jobs', {'id': job_id})
            job['values'] = result
            job['status'] = 'complete'
            job['end time'] = end_time
            db.save(job, experiment_name, 'jobs', {'id': job_id})

        else:
            sys.stderr.write("Job failed in %0.2f seconds.\n" %
                             (end_time - start_time))

            # Update metadata.
            job = db.load(experiment_name, 'jobs', {'id': job_id})
            job['status'] = 'broken'
            job['end time'] = end_time
            db.save(job, experiment_name, 'jobs', {'id': job_id})

        return
    elif validation is True and elc is False:
        success = False
        try:
            if job['language'].lower() == 'python':
                job = db.load(experiment_name, 'jobs', {'id': job_id})
                result = python_validation_accs(job)
                if not isinstance(result, dict):
                    if np.any(np.isnan(result)):
                        sys.stderr.write('Atleast one of the results has a nan\n')
                        
                        r = [np.nan] * len(result)
                        rs = [np.array(r) for t in job['tasks']]
                        result = dict(zip(job['tasks'], rs))
                    elif len(job['tasks']) == 1: # Only one named job
                        result = {job['tasks'][0]: result}
                    else:
                        result = {'main': result}
                if set(result.keys()) != set(job['tasks']):
                    raise Exception('Result task names {} did not match job task names {}'.format(
                        result.keys(), job['tasks']))
                
                prev_result = job['validation_accs']
                
                updated = False
                if prev_result is None:
                    updated = True
                else:
                    
                    for k in prev_result.keys():
                        if not k in result:
                            raise Exception('Key {} not found in result'.format(k))
                        p_v = prev_result[k]
                        c_v = result[k]
                        if len(p_v) < len(c_v):
                            updated = True

                if updated is True:
                    # New validation accs have been found
                    job = db.load(experiment_name, 'jobs', {'id': job_id})
                    job['validation_accs'] = result
                    job['validation_updated'] = True
                    db.save(job, experiment_name, 'jobs', {'id': job_id})
            else:
                raise Exception("This language has not been implemented")

            success = True
        except:
            import traceback
            traceback.print_exc()
            sys.stderr.write("Problem getting validation results\n")
            print sys.exc_info()

        if success:
            sys.stderr.write(
                "Completed getting validation results for job %d.\n" % job_id)
        else:
            sys.stderr.write("Failed to get validation results for job %d.\n"
                             % job_id)
            job = db.load(experiment_name, 'jobs', {'id': job_id})
            job['validation_updated'] = False
            db.save(job, experiment_name, 'jobs', {'id': job_id})
        return
    elif validation is False and elc is True:
        if prev_jobs is None:
            prev_jobs = []
        if isinstance(prev_jobs, dict):
            prev_jobs = [prev_jobs]
        success = False
        try:
            if job['language'].lower() == 'python':
                ret_dict = python_elc(job,
                                      mode,
                                      prob_x_greater_type,
                                      threshold,
                                      predictive_std_threshold,
                                      nthreads,
                                      xlim,
                                      prev_jobs=prev_jobs,
                                      min_y_prev=min_y_prev)
                job = db.load(experiment_name, 'jobs', {'id': job_id})
                job['elc_result'] = ret_dict
                job['elc_status'] = 'complete'
                db.save(job, experiment_name, 'jobs', {'id': job_id})
                success = True
            else:
                raise Exception("Not implemented for this language")
        except:
            import traceback
            traceback.print_exc()
            sys.stderr.write("Problem getting extrapolation running\n")
            print sys.exc_info()
            job = db.load(experiment_name, 'jobs', {'id': job_id})
            job['elc_status'] = 'broken'
            db.save(job, experiment_name, 'jobs', {'id': job_id})
        if success:
            sys.stderr.write(
                "Completed getting extrapolation for job %d.\n" % job_id)
        else:
            sys.stderr.write("Failed to get extrapolation for job %d.\n"
                             % job_id)
    else:
        sys.stderr.write("Error: both validation and ELC specified as true.\n")
    return


def python_launcher(job):
    # Run a Python function
    sys.stderr.write("Running python job.\n")

    # Add directory to the system path.
    sys.path.append(os.path.realpath(job['expt_dir']))

    # Change into the directory.
    os.chdir(job['expt_dir'])
    sys.stderr.write("Changed into dir %s\n" % (os.getcwd()))

    # Convert the JSON object into useful parameters.
    params = {}
    for name, param in job['params'].iteritems():
        vals = param['values']

        if param['type'].lower() == 'float':
            params[name] = np.array(vals)
        elif param['type'].lower() == 'int':
            params[name] = np.array(vals, dtype=int)
        elif param['type'].lower() == 'enum':
            params[name] = vals
        else:
            raise Exception("Unknown parameter type.")

    # Load up this module and run
    main_file = job['main-file']
    if main_file[-3:] == '.py':
        main_file = main_file[:-3]
    sys.stderr.write('Importing %s.py\n' % main_file)
    module = __import__(main_file)
    sys.stderr.write('Running %s.main()\n' % main_file)
    result = module.main(job['id'], params)

    # Change back out.
    os.chdir('..')

    # TODO: add dict capability

    sys.stderr.write("Got result %s\n" % (result))

    return result


def python_validation_accs(job):
    """submit a python job to get validation accuracies over epochs

    This function is used to get the intermediate stages of the current job.
    It will primarily get the validation accuracies in the form of a numpy list

    Arguments:
        job {[type]} -- [description]

    Returns:
        [type] -- [description]

    Raises:
        Exception -- [description]
    """
    sys.stderr.write("Checking python job.\n")

    # Add directory to the system path.
    sys.path.append(os.path.realpath(job['expt_dir']))

    # Change into the directory.
    os.chdir(job['expt_dir'])
    sys.stderr.write("Changed into dir %s\n" % (os.getcwd()))

    # Convert the JSON object into useful parameters.
    params = {}
    for name, param in job['params'].iteritems():
        vals = param['values']

        if param['type'].lower() == 'float':
            params[name] = np.array(vals)
        elif param['type'].lower() == 'int':
            params[name] = np.array(vals, dtype=int)
        elif param['type'].lower() == 'enum':
            params[name] = vals
        else:
            raise Exception("Unknown parameter type.")

    # Load up this module and run
    main_file = job['main-file']
    if main_file[-3:] == '.py':
        main_file = main_file[:-3]
    sys.stderr.write('Importing %s.py\n' % main_file)
    module = __import__(main_file)
    sys.stderr.write('Running %s.get_validation_accuracies()\n' % main_file)
    validation_accs = module.get_validation_accuracies(job['id'], params)

    # Change back out.
    os.chdir('..')

    # TODO: add dict capability

    sys.stderr.write("Got result: {}.\n".format(validation_accs))
    # TODO: check if the results are a numpy array
    return validation_accs


def python_elc(job, mode, prob_x_greater_type, threshold,
               predictive_std_threshold, nthreads, xlim, prev_jobs = [], min_y_prev=1,
               recency_weighting=False, monotonicity_condition=False, task=None):
    if task is None:
        task = job['tasks'][0]
    y_list = np.array(job['validation_accs'][task])
    y_best = job['y_best']
    y_prev_ids =[j['id'] for j in prev_jobs]
    y_prev_list = [np.array(j['validation_accs'][task]) for j in prev_jobs]
    y_covs_list = job['cov_list']
    y_cov_ids = job['cov_ids']
    y_cov_list = [y_covs_list[y_cov_ids.index(jid)] for jid in y_prev_ids]
    if mode == 'conservative':
        term_crit = ConservativeTerminationCriterion(
            y_list, xlim, nthreads, prob_x_greater_type,
            predictive_std_threshold=predictive_std_threshold)
    elif mode == 'optimistic':
        term_crit = OptimisticTerminationCriterion(
            y_list, xlim, nthreads, prob_x_greater_type,
            predictive_std_threshold=predictive_std_threshold)
    elif mode == 'gelc-conservative':
        term_crit = GelcConservativeTerminationCriterion(
            y_list, xlim,
            prob_x_greater_type=prob_x_greater_type,
            y_prev_list=y_prev_list,
            predictive_std_threshold=predictive_std_threshold,
            min_y_prev=min_y_prev,
            recency_weighting=recency_weighting,
            monotonicity_condition=monotonicity_condition)
    elif mode == 'gelc-hp-conservative':
        term_crit = GelcHpConservativeTerminationCriterion(
            y_list, xlim,
            prob_x_greater_type=prob_x_greater_type,
            y_prev_list=y_prev_list,
            predictive_std_threshold=predictive_std_threshold,
            min_y_prev=min_y_prev,
            recency_weighting=recency_weighting,
            monotonicity_condition=monotonicity_condition,
            y_cov_list=y_cov_list)
    else:
        raise Exception("Invalid mode for ELC: {}".format(mode))
    result = term_crit.run(y_best, threshold=threshold)
    return result

# # BROKEN
# def matlab_launcher(job):
#     # Run it as a Matlab function.

#     try:
#         import pymatlab
#     except:
#         raise Exception("Cannot import pymatlab. pymatlab is required for \
#             Matlab jobs. It is installable with pip.")

#     sys.stderr.write("Booting up Matlab...\n")
#     session = pymatlab.session_factory()

#     # Add directory to the Matlab path.
#     session.run("cd('%s')" % os.path.realpath(job['expt_dir']))

#     session.run('params = struct()')
#     for name, param in job['params'].iteritems():
#         vals = param['values']

#         # sys.stderr.write('%s = %s\n' % (param['name'], str(vals)))

#         # should have dtype=float explicitly, otherwise
#         # if they are ints it will automatically do int64, which
#         # matlab will receive, and will tend to break matlab scripts
#         # because in matlab things tend to always be double type
#         session.putvalue('params_%s' % name, np.array(vals, dtype=float))
#         session.run("params.%s = params_%s" % (name, name))
#         # pymatlab sucks, so I cannot put the value directly into a struct
#         # instead i do this silly workaround to put it in a variable and then
#         # copy that over into the struct
#         # session.run('params_%s'%param['name'])

#     sys.stderr.write('Running function %s\n' % job['function-name'])

#     # Execute the function
#     session.run('result = %s(params)' % job['function-name'])

#     # Get the result
#     result = session.getvalue('result')

#     # TODO: this only works for single-task right now
#     result = float(result)
#     sys.stderr.write("Got result %s\n" % (result))

#     del session

#     return result


# # BROKEN
# def shell_launcher(job):
#     # Change into the directory.
#     os.chdir(job['expt_dir'])
#     cmd = './%s %s' % (job['function-name'], job_file)
#     sys.stderr.write("Executing command '%s'\n" % cmd)

#     subprocess.check_call(cmd, shell=True)
#     return result


# # BROKEN
# def mcr_launcher(job):
#     # Change into the directory.
#     os.chdir(job['expt_dir'])

#     if 'MATLAB' in os.environ:
#         mcr_loc = os.environ['MATLAB']
#     else:
#         raise Exception("Please set the MATLAB environment variable")

#     cmd = './run_%s.sh %s %s' % (job['function-name'], mcr_loc, job_file)
#     sys.stderr.write("Executing command '%s'\n" % (cmd))
#     subprocess.check_call(cmd, shell=True)

#     return result

if __name__ == '__main__':
    main()
