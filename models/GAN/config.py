import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR = 3e-4            # Learning rate for Adam
Z_DIM = 64           # Dimensionality of input noise vector

IMAGE_SIZE = 64      # Generated \ Input images dimension
IMAGE_CHANNELS = 3   # Number of color channels i trainig images
INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS # [Height * Width * color_channels]

BATCH_SIZE = 32      # DataLoader Batch size
NUM_EPOCHS = 50      # Epoch per training loop

# Fixed noise for tensorboard predictions
torch.manual_seed(2003)
FIXED_NOISE = torch.rand((BATCH_SIZE, Z_DIM)).to(DEVICE)

# Paths to different directories (data, cached weights, tensorboard logs)
DATASET = "data/pokemon/raw"
CLASSES_DIR = "data/Pokemon.txt"
CHECKPOINT_GEN = "models/GAN/cached/generator.pth"
CHECKPOINT_CRITIC = "models/GAN/cached/critic.pth"
LOG_DIR = "models/GAN/logs"

# Flags to save\load the model
LOAD_MODEL=True
SAVE_MODEL=True