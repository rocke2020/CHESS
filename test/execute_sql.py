import argparse
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from datasketch import MinHash, MinHashLSH
from icecream import ic
from loguru import logger
from pandas import DataFrame
from torch import nn
from torch.utils import data
from tqdm import tqdm

sys.path.append(os.path.abspath('.'))
from src.database_utils.execution import execute_sql

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEED = 42
random.seed(SEED)

db_root = '/data/t2s-data/dataset/spider/spider-fixed/spider/database'
unique_values_in_db = {
    "table1": {
        "column1": ["value1", "value2", "value3"],
        "column2": ["value1", "value2", "value4"],
    }
}
