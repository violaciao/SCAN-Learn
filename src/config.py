import os

# WORKING_DIR = "/Users/Viola/CDS/AAI/Project/SCAN-Learn"
WORKING_DIR = "/scratch/xc965/DL/SCAN-Learn"

MODEL_VERSION = "016"

EMBEDDEING_PRETRAINED = False
WEIGHT_UPDATE = True
EMBEDDEING_SOURCE = "google"
HIDDEN_SIZE = 300
TEACHER_FORCING_RATIO = 0.5
EVAL_TRNorTST = "train"
LEARNING_RATE = 0.01
MAX_LENGTH = 50
N_ITERS = 75000
TASK_NAME = "addprim-jump"

if not os.path.exists("saved_models"):
	os.makedirs("saved_models")
