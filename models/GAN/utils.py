import torch
import random
import numpy as np
import os
import torchvision
import torch.nn as nn
import config
from torchvision.utils import save_image
from scipy.stats import truncnorm

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_examples(gen, model_name="others", truncation=0.7, n=100):
    """
    Tried using truncation trick here but not sure it actually helped anything, you can
    remove it if you like and just sample from torch.randn
    """
    print("=> Generating Examples")
    gen.eval()
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, config.Z_DIM).to(config.DEVICE)
            img = gen(noise)
            img = img.reshape(-1, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
            save_image(img*0.5+0.5, f"images/{model_name}/img_{i}.png")
    gen.train()