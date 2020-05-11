import errno
import os

def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def prefix_dict(d, prefix):
    return {'/'.join([prefix, k]): v for k, v in d.items()}

