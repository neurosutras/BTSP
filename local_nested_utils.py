import math
import datetime
import copy
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import scipy.optimize
import scipy.signal as signal
import scipy.stats as stats
import random
import pprint
import sys
import os
import gc
import importlib
import traceback
import collections
from collections import Iterable, defaultdict

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['text.usetex'] = False


class Context(object):
    """
    A container replacement for global variables to be shared and modified by any function in a module.
    """
    def __init__(self, namespace_dict=None, **kwargs):
        self.update(namespace_dict, **kwargs)

    def update(self, namespace_dict=None, **kwargs):
        """
        Converts items in a dictionary (such as globals() or locals()) into context object internals.
        :param namespace_dict: dict
        """
        if namespace_dict is not None:
            self.__dict__.update(namespace_dict)
        self.__dict__.update(kwargs)

    def __call__(self):
        return self.__dict__

    def __getitem__(self, key):
        return self.__dict__[key]


def list_find(f, items):
    """
    Return index of first instance that matches criterion.
    :param f: callable
    :param items: list
    :return: int
    """
    for i, x in enumerate(items):
        if f(x):
            return i
    return None


def get_unknown_click_arg_dict(cli_args):
    """

    :param cli_args: list of str: contains unknown click arguments as list of str
    :return: dict
    """
    kwargs = {}
    for arg in cli_args:
        arg_split = arg.split('=')
        key = arg_split[0][2:]
        if len(arg_split) < 2:
            val = True
        else:
            val = arg_split[1]
        kwargs[key] = val
    return kwargs


def read_from_yaml(file_path, Loader=None):
    """
    Import a python dict from .yaml
    :param file_path: str (should end in '.yaml')
    :param Loader: :class:'yaml.Loader'
    :return: dict
    """
    import yaml
    if Loader is None:
        Loader = yaml.FullLoader
    if os.path.isfile(file_path):
        with open(file_path, 'r') as stream:
            data = yaml.load(stream, Loader=Loader)
        return data
    else:
        raise Exception('File: {} does not exist.'.format(file_path))


def clean_axes(axes, left=True, right=False):
    """
    Remove top and right axes from pyplot axes object.
    :param axes: list of pyplot.Axes
    :param top: bool
    :param left: bool
    :param right: bool
    """
    if not type(axes) in [np.ndarray, list]:
        axes = [axes]
    elif type(axes) == np.ndarray:
        axes = axes.flatten()
    for axis in axes:
        axis.tick_params(direction='out')
        axis.spines['top'].set_visible(False)
        if not right:
            axis.spines['right'].set_visible(False)
        if not left:
            axis.spines['left'].set_visible(False)
        axis.get_xaxis().tick_bottom()
        axis.get_yaxis().tick_left()


def get_h5py_group(file, hierarchy, create=False):
    """

    :param file: :class: in ['h5py.File', 'h5py.Group']
    :param hierarchy: list of str
    :param create: bool
    :return: :class:'h5py.Group'
    """
    target = file
    for key in hierarchy:
        if key is not None:
            key = str(key)
            if key not in target:
                if create:
                    target = target.create_group(key)
                else:
                    raise KeyError('get_h5py_group: target: %s does not contain key: %s; valid keys: %s' %
                                   (target, key, list(target.keys())))
            else:
                target = target[key]
    return target


def get_h5py_attr(attrs, key):
    """
    str values are stored as bytes in h5py container attrs dictionaries. This function enables py2/py3 compatibility by
    always returning them to str type upon read. Values should be converted during write with the companion function
    set_h5py_str_attr.
    :param attrs: :class:'h5py._hl.attrs.AttributeManager'
    :param key: str
    :return: val with type converted if str or array of str
    """
    if key not in attrs:
        raise KeyError('get_h5py_attr: invalid key: %s' % key)
    val = attrs[key]
    if isinstance(val, basestring):
        val = np.string_(val).astype(str)
    elif isinstance(val, Iterable) and len(val) > 0:
        if isinstance(val[0], basestring):
            val = np.array(val, dtype='str')
    return val


def set_h5py_attr(attrs, key, val):
    """
    str values are stored as bytes in h5py container attrs dictionaries. This function enables py2/py3 compatibility by
    always converting them to np.string_ upon write. Values should be converted back to str during read with the
    companion function get_h5py_str_attr.
    :param attrs: :class:'h5py._hl.attrs.AttributeManager'
    :param key: str
    :param val: type converted if str or array of str
    """
    if isinstance(val, basestring):
        val = np.string_(val)
    elif isinstance(val, Iterable) and len(val) > 0:
        if isinstance(val[0], basestring):
            val = np.array(val, dtype='S')
    attrs[key] = val