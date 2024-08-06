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


def test_execute_sql():
    for db_name in os.listdir(db_root):
        # yelp academic
        if db_name != "academic":
            continue
        db_file = f"{db_root}/{db_name}/{db_name}.sqlite"
        sql = "SELECT name FROM sqlite_master WHERE type='table';"
        logger.info(f"start {db_name}")

        r = execute_sql(db_file, sql)
        for table in r:
            logger.info(table)


def check_unique_values(unique_values):
    total_unique_values = sum(
        len(column_values) for table_values in unique_values.values() for column_values in table_values.values()
    )
    ic(total_unique_values)


def test_minhash():
    data1 = [
        "minhash",
        "is",
        "a",
        "probabilistic",
        "data",
        "structure",
        "for",
        "estimating",
        "the",
        "similarity",
        "between",
        "datasets",
    ]
    data2 = [
        "minhash",
        "is",
        "a",
        "probability",
        "data",
        "structure",
        "for",
        "estimating",
        "the",
        "similarity",
        "between",
        "documents",
    ]

    m1, m2 = MinHash(), MinHash()
    for d in data1:
        m1.update(d.encode("utf8"))
    for d in data2:
        m2.update(d.encode("utf8"))
    print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))

    s1 = set(data1)
    s2 = set(data2)
    actual_jaccard = float(len(s1.intersection(s2))) / float(len(s1.union(s2)))
    print("Actual Jaccard for data1 and data2 is", actual_jaccard)


def test_minhash_lsh():
    set1 = set(
        [
            "minhash",
            "is",
            "a",
            "probabilistic",
            "data",
            "structure",
            "for",
            "estimating",
            "the",
            "similarity",
            "between",
            "datasets",
        ]
    )
    set2 = set(
        [
            "minhash",
            "is",
            "a",
            "probability",
            "data",
            "structure",
            "for",
            "estimating",
            "the",
            "similarity",
            "between",
            "documents",
        ]
    )
    set3 = set(
        [
            "minhash",
            "is",
            "probability",
            "data",
            "structure",
            "for",
            "estimating",
            "the",
            "similarity",
            "between",
            "documents",
        ]
    )

    m1 = MinHash(num_perm=128)
    m2 = MinHash(num_perm=128)
    m3 = MinHash(num_perm=128)
    for d in set1:
        m1.update(d.encode("utf8"))
    for d in set2:
        m2.update(d.encode("utf8"))
    for d in set3:
        m3.update(d.encode("utf8"))

    # Create LSH index
    lsh = MinHashLSH(threshold=0.5, num_perm=128)
    lsh.insert("m2", m2)
    lsh.insert("m3", m3)
    result = lsh.query(m1)
    print("Approximate neighbours with Jaccard similarity > 0.5", result)
    file = "test_lsh.pkl"
    with open(file, "wb") as f:
        pickle.dump(lsh, f)
    with open(file, "rb") as f:
        lsh = pickle.load(f)


# test_minhash()
# test_minhash_lsh()
# check_unique_values(unique_values_in_db)
with open("test/test_lsh.pkl", "rb") as f:
    lsh = pickle.load(f)
