import os

# WORKING_DIR = "/Users/Viola/CDS/AAI/Project/SCAN-Learn"
WORKING_DIR = "/scratch/xc965/DL/SCAN-Learn"
HIDDEN_SIZE = 100
TEACHER_FORCING_RATIO = 0.5
MODEL_VERSION = "012"
LEARNING_RATE = 0.01
MAX_LENGTH = 50
N_ITERS = 75000
TASK_NAME = "addprim-jump"

if not os.path.exists("saved_models"):
	os.makedirs("saved_models")
