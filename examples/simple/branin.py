import math
import time

import numpy as np


def branin(x, y):

    result = np.square(y - (5.1 / (4 * np.square(math.pi))) * np.square(x) +
                       (5 / math.pi) * x - 6) + 10 * (1 - (1. / (8 * math.pi))
                                                      ) * np.cos(x) + 10

    result = float(result)

    print 'Result = %f' % result

    return result


# Write a function like this called 'main'
def main(job_id, params):
    time.sleep(5)
    print 'Anything printed here will end up in the \
    output directory for job #%d' % job_id
    print params
    return branin(params['x'], params['y'])


def get_validation_accuracies(job_id, params):
    print 'getting validation accs'
    return np.array([job_id])
