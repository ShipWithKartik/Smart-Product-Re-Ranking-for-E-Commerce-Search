"""
Data Extraction Module for Smart Product Re-Ranking System
Provides database connection utilities and SQL query execution with error handling
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import mysql.connector
from mysql.connector import Error as MySQLError
import logging
from typing import Dict, Any, Optional, List, Tuple
import time
from contextlib import contextmanager
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logg