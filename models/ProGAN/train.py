import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm 
import config
from torch.utils.data import Dataset
from typing import Tuple, Dict, List
import re
from pathlib import Path 
import os
import pandas as pd
from PIL import Image

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

torch.backends.cudnn.benchmark = True

def get_classes(df: pd.DataFrame):
    classes = df["Name"].unique()
    classes_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, classes_to_idx

class PokemonData(Dataset):
    def __init__(self,
                 targ_dir: str,
                 classes_df: pd.DataFrame,
                 transform=None):
        super().__init__()
        self.paths = list(Path(targ_dir).glob('*.png'))
        self.transform = transform
        self.classes, self.classes_to_idx = get_classes(classes_df)
        
    def load_image(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path = self.paths[index]
        image = Image.open(image_path).convert("RGBA")
        return Image.composite(image, Image.new('RGBA', image.size, 'white'), image).convert("RGB")
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index: int):
        img = self.load_image(index)
        img_name = self.paths[index].name.split('.')[0]
        img_name = 3
        class_name = self.classes[min(720, img_name)]
        class_idx = self.classes_to_idx[class_name]
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx


pokemon_data = pd.read_csv("data/Pokemon.txt")
pokemon_data = pokemon_data.drop_duplicates(subset=["#"],keep="first")
pokemon_data = pokemon_data.set_index(["#"], drop=True)

def get_loader(img_size):
    transform = transforms.Compose([
        transforms.Resize(((img_size, img_size))),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(
            [0.5 for _ in range(config.CHANNELS_IMG)],
            [0.5 for _ in range(config.CHANNELS_IMG)]
        )
    ])
    batch_size = config.BATCH_SIZES[int(log2(img_size / 4))]
    dataset = PokemonData(targ_dir=config.DATASET, transform=transform, classes_df=pokemon_data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset

def train_fn(critic, 
             gen,
             loader,
             dataset,
             step,
             alpha,
             opt_critic,
             opt_gen,
             tensorboard_step,
             writer,
             scaler_gen,
             scaler_critic):
    loop = tqdm(loader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]
        
        # Train Critic max (E[critic(real)] - E[critic(fake)])
        noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)
        
        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real**2))
            )
            
        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()
        
        # Train generator: max E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)
            
        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()
        
        alpha += cur_batch_size / (len(dataset) * config.PROGRESSIVE_EPOCHS[step]*0.5)
        alpha = min(alpha, 1)
        
        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step
            )
            tensorboard_step += 1
            
    return tensorboard_step, alpha
        

def main():
    gen = Generator(config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    critic = Discriminator(config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(config.DEVICE)
    
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    
    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_critic = torch.cuda.amp.GradScaler()
    
    writer = SummaryWriter(f"logs/gan")
    
    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE)

    gen.train()
    critic.train()
    
    tensorboard_step = 0
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:]:
        alpha = 1e-5
        loader, dataset = get_loader(4*2**step)
        print(f"Image size: {4*2**step}")
        for epoch in range(num_epochs):
            if config.GENERATE_EXAMPLES:
                generate_examples(gen, steps=step, model_name="ProGAN")
                
            print(f"Epoch: [{epoch}/{num_epochs}]")
            tensorboard_step, alpha = train_fn(
                critic, 
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_critic
            )
            
            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC)

        step += 1
    
if __name__=="__main__":
    main()