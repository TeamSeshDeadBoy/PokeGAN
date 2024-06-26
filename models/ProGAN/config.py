# import cv2
import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 32
DATASET = 'data/pokemon/raw'
CHECKPOINT_GEN = "models/ProGan/cached/generator.pth"
CHECKPOINT_CRITIC = "models/ProGan/cached/critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = True
LEARNING_RATE = 1e-3
BATCH_SIZES = [16, 16, 16, 16, 16, 16, 16, 8, 4]
IMAGE_SIZE = 128
CHANNELS_IMG = 3
Z_DIM = 256  # should be 512 in original paper
IN_CHANNELS = 256  # should be 512 in original paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [10] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4
GENERATE_EXAMPLES = True
LOGS_DIR = "models/ProGAN/logs"