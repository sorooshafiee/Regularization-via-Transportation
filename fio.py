""" Access layer for File I/O """

import pickle


# ------------------------------------------------------------------------------
# File pickling in order to cache or decache
# ------------------------------------------------------------------------------
DIR_CACHE = './datacache/'


def print_count(count, msg=''):
    """ print file info """
    try:
        print(str(locale.format("%d", count, grouping=True)).rjust(12)
              + '  ' + msg.strip())
    except:
        print(str(count).rjust(12) + '  ' + msg.strip())


def cache(obj, name, count=None, msg='', silent=False):
    """ cache a file """
    with open(DIR_CACHE + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    if not count:
        count = len(obj)
    if not msg:
        msg = name
    if not silent:
        print_count(count, msg)


def decache(name, msg='', silent=False):
    """ decache a file """
    try:
        with open(DIR_CACHE + name + '.pkl', 'rb') as f:
            obj = pickle.load(f)
        if not msg:
            msg = name
        msg = str(msg).ljust(30) + ' (loaded from cache)'
        if not silent:
            print_count(len(obj), msg)
    except:
        obj = None
    return obj

