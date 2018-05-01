import os

WORKING_DIR = "/Users/Viola/CDS/AAI/Project/SCAN-Learn"
# WORKING_DIR = "/scratch/xc965/DL/SCAN-Learn"

MODEL_VERSION = "023"

EMBEDDING_PRETRAINED = True
WEIGHT_UPDATE = False
EMBEDDEING_SOURCE = "glove"
HIDDEN_SIZE = 300
EVAL_TRNorTST = "test"
BASE_TF_RATIO = 0.8
BASE_LR = 0.0001
DECAY_WEIGHT = 0.5
ITER_DECAY = 10000
MAX_LENGTH = 50
N_ITERS = 75000
TASK_NAME = "addprim-jump"

if not os.path.exists("saved_models"):
	os.makedirs("saved_models")
