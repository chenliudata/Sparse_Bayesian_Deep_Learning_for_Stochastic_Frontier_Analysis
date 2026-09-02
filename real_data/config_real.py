
import math

# MCMC settings 
NUM_SAMPLES = 1000
WARMUP_STEPS = 1000
NUM_CHAINS = 2
TARGET_ACCEPT_PROB = 0.8
MAX_TREE_DEPTH = 10

# Inverse-Gamma prior calibration
IG_DF = 3
IG_Q = 0.95
ALPHA_U = IG_DF / 2
ALPHA_V = IG_DF / 2

# Weight prior scales 
PRIOR_VAR = 1
SCALE_NORMAL = math.sqrt(PRIOR_VAR)
SCALE_HALFNORMAL = math.sqrt(PRIOR_VAR / (1.0 - 2.0 / math.pi))
SCALE_HALFLAPLACE = math.sqrt(PRIOR_VAR)
SCALE_LAPLACE = math.sqrt(PRIOR_VAR / 2)

# Neural network architecture
N_HIDDEN1 = 16
N_HIDDEN2 = 8

# NN-MLE warm-start settings
MLE_NN_EPOCHS = 3000
MLE_NN_LR = 0.001

# Real data
DATA_PATH = "rice92_data.pkl"
OUTPUT_VAR = "PROD"
INPUT_VARS = ["AREA", "LABOR", "NPK"]
N_INPUTS = len(INPUT_VARS)

# Output directory
BASE_RESULTS = "results_rice92_two_chains"
