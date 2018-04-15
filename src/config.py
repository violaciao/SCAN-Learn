import os

WORKING_DIR = "/Users/Viola/CDS/AAI/Project/SCAN-Learn"

MAX_LENGTH = 40
MODEL_VERSION = "005"
LEARNING_RATE = 0.01
N_ITERS = 10
TASK_NAME = "data/processed/train-addprim-jump"
TEST_SENTENCE = "run twice after jump opposite left"

if not os.path.exists("saved_models"):
	os.makedirs("saved_models")
