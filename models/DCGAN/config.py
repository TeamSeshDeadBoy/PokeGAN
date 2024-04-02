import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


LR = 2e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
Z_DIM = 100
NUM_EPOCHS = 40
FEATURE_DISC = 64
FEATURE_GEN = 64

# Fixed noise for tensorboard predictions
torch.manual_seed(2003)
FIXED_NOISE = torch.rand((BATCH_SIZE, Z_DIM, 1, 1)).to(DEVICE)

# Paths to different directories (data, cached weights, tensorboard logs)
DATASET = "data/pokemon/raw"
CLASSES_DIR = "data/Pokemon.txt"
CHECKPOINT_GEN = "models/DCGAN/cached/generator.pth"
CHECKPOINT_CRITIC = "models/DCGAN/cached/critic.pth"
LOG_DIR = "models/DCGAN/logs"

# Toggles
LOAD_MODEL = True
SAVE_MODEL = True