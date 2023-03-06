import string
import logging
import random

import numpy as np
from IPython import embed
from datetime import datetime

from hotline.hotline import *

logging.basicConfig(format='%(asctime)s.%(msecs)d[%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)
log = logging.getLogger('main')


def decorator(func):
    def _decorator(self, *args, **kwargs):
        log.info(f'Begin {func.__name__}')
        dt1 = datetime.now()

        # Do main function
        ret = func(self, *args, **kwargs)

        dt2 = datetime.now()
        tdelta = dt2 - dt1
        self.fn_runtimes.append({'name': func.__name__, 'time': tdelta})
        log.info(f'End {func.__name__} [runtime: {tdelta}]')

        return ret
    return _decorator

class Wrapper(object):
    """Usage examples:
    with Wrapper('load_raw_trace', self.fn_runtimes):
        h_read.load_raw_trace(self.trace_filepath)
    """
    def __init__(self, name, fn_runtimes):
        self.name = name
        self.fn_runtimes = fn_runtimes

    def __enter__(self):
        log.info(f'Begin {self.name}')
        self.dt1 = datetime.now()

    def __exit__(self, a, b, c):
        dt2 = datetime.now()
        tdelta = dt2 - self.dt1
        self.fn_runtimes.append({'name': self.name, 'time': tdelta})
        log.info(f'End {self.name} [runtime: {tdelta}]')

def normalize_0_to_1(data):
  data = np.array(data)
  return (data - np.min(data)) / (np.max(data) - np.min(data))

def random_id(length=16):
  return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
