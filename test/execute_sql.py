from pathlib import Path
import os, sys, shutil, json
from datetime import datetime
from collections import defaultdict
import re, random, math, logging
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import pandas as pd
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from icecream import ic
from loguru import logger

sys.path.append(os.path.abspath('.'))
from src.database_utils.execution import execute_sql

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEED = 42
random.seed(SEED)

db_root = '/data/t2s-data/dataset/spider/spider-fixed/spider/database'

for db_name in os.listdir(db_root):
    # yelp academic
    if db_name != 'academic':
        continue
    db_file = f'{db_root}/{db_name}/{db_name}.sqlite'
    sql = "SELECT name FROM sqlite_master WHERE type='table';"
    logger.info(f'start {db_name}')

    r = execute_sql(db_file, sql)
    for table in r:
        logger.info(table)
