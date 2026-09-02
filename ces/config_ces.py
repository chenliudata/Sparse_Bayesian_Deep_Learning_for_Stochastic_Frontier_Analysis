import math

# MCMC settings
NUM_SAMPLES = 1000
WARMUP_STEPS = 1000
NUM_CHAINS = 1
TARGET_ACCEPT_PROB = 0.8
MAX_TREE_DEPTH = 10

# True CES DGP parameters.
CES_PARAMS_TRUE = {
    "log_A": 1.0,
    "delta": 0.5,
    "rho": -0.5,
    "nu": 0.8,
}

SIGMA_U_TRUE = 0.5
SIGMA_V_TRUE = 0.8

# Sample size
N = 50

# Inverse-Gamma prior calibration
IG_DF = 3
IG_Q = 0.5

ALPHA_U = IG_DF / 2
ALPHA_V = IG_DF / 2
BETA_U = None
BETA_V = None

# Weight prior scales
PRIOR_VAR = 1.0
SCALE_NORMAL = math.sqrt(PRIOR_VAR)
SCALE_HALFNORMAL = math.sqrt(PRIOR_VAR / (1.0 - 2.0 / math.pi))
SCALE_HALFLAPLACE = math.sqrt(PRIOR_VAR)
SCALE_LAPLACE = math.sqrt(PRIOR_VAR / 2)

# Neural network architecture
N_HIDDEN1 = 16
N_HIDDEN2 = 8

# NN-MLE warm-start settings
MLE_RELU_EPOCHS = 3000
MLE_CRELU_EPOCHS = 500
MLE_NN_LR = 0.001

# Output and replication settings
SEEDS = list(range(1, 101))
BASE_RESULTS = f"results_ces_N{N}_78"
