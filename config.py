import json

__json = json.load(open('config.json', 'r'))

DATASET_DIR = __json.get('dataset_dir')
TEST_CSV = __json.get('test_csv')
TRAIN_CSV = __json.get('train_csv')
SUBMISSION_CSV = __json.get('submission_csv')

TEST_SIZE = __json.get('test_size')
BATCH_SIZE = __json.get('batch_size')

SEED = 7
mu = 398600.44180000003
real_columns = ['x', 'y', 'z', 'Vx', 'Vy', 'Vz']
sim_columns = ['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']
