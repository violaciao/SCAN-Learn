import os

# WORKING_DIR = "/Users/Viola/CDS/AAI/Project/SCAN-Learn"
WORKING_DIR = "/scratch/xc965/DL/SCAN-Learn"
HIDDEN_SIZE = 50
MAX_LENGTH = 100
MODEL_VERSION = "009"
LEARNING_RATE = 0.01
N_ITERS = 75000
TASK_NAME = "addprim-jump"

if not os.path.exists("saved_models"):
	os.makedirs("saved_models")
