import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_PATH = Path(__file__).parent.parent.parent

# 目录
DATA_DIR = ROOT_PATH / 'data'
NER = 'ner'
LOGS_DIR = ROOT_PATH / 'logs'
CHECKPOINTS_DIR = ROOT_PATH / 'checkpoints'
# 环境变量
ENV_PATH = str(ROOT_PATH / '.env')

# 数据相关
NER_RAW_DATA_PATH = DATA_DIR / NER / 'raw' / 'data.json'
NER_PROCESSED_DATA_PATH = DATA_DIR / NER / 'processed'
MODEL_NAME = 'google-bert/bert-base-chinese'

# 超参数
BATCH_SIZE = 2
EPOCH = 5
SAVE_STEP = 20
LEARNING_RATE = 5e-5
LABELS = ['B', 'I', 'O']

# 数据库相关
load_dotenv(ENV_PATH)
MySQL_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': os.getenv("MYSQL_PASSWORD"),
    'database': 'gmall'
}
# 图数据库
NEO4J_CONFIG = {
    'uri': 'neo4j://localhost:7687',
    'auth': ('neo4j', os.getenv("NEO4J_PASSWORD"))
}
