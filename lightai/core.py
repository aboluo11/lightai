from IPython.core.debugger import set_trace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import sys
from pathlib import Path
from itertools import chain
import re
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import OrderedDict
import os
from tensorboardX import SummaryWriter
from typing import Callable, Optional, Collection, Any, Type, NamedTuple, List, Iterable, Sequence
import pickle
import dill
from functools import partial
import cv2
from fastprogress import master_bar, progress_bar
from glob import glob


def apply_leaf(module, func, **kwargs):
    childs = list(module.children())
    if len(childs) == 0:
        func(module, **kwargs)
        return
    for child in childs:
        apply_leaf(child, func, **kwargs)
