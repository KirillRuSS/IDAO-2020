import json


__json = json.load(open('config.json', 'r'))

DATASET_DIR: __json.get('dataset_dir')
TEST_PATH = __json.get('test_path')
TRAIN_PATH = __json.get('train_path')
SUBMISSION_CVS = __json.get('submission_csv')

TEST_SIZE = __json.get('test_size')
BATCH_SIZE = __json.get('batch_size')

SEED = 7