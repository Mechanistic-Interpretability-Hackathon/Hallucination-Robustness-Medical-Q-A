# Standard library imports
import os
import json
import ast
import re
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import numpy as np
import pandas as pd
import scipy.stats
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    cohen_kappa_score
)
import goodfire

# Logging configuration
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)