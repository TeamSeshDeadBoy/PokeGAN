from model import Discriminator, Generator
import config
import torch
from torchvision import transforms
import data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision

# Instantiating a Discriminator and a Generator models with config hyperparamethers, sending them to target Device
disc = Discriminator(config.INPUT_SIZE).to(config.DEVICE)
gen = Generator(config.Z_DIM, config.INPUT_SIZE).to(config.DEVICE)

# Setting up a transformer for Pokemon Data
tranformer = transforms.Compose([
    transforms.Resize(((config.IMAGE_SIZE, config.IMAGE_SIZE))),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(
        [0.5 for _ in range(config.IMAGE_CHANNELS)],
        [0.5 for _ in range(config.IMAGE_CHANNELS)]
    )
])

# Create a Dataset Instance from custom data class
dataset = data.PokemonData(targ_dir=config.DATASET, classes_df_dir=config.CLASSES_DIR, transform=tranformer)

# Creating a DataLoader objects with configured Dataclass and a config batch size
pokemon_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True)

# tensorboard logging
writer_fake = SummaryWriter(config.LOG_DIR + "/fake")
writer_real = SummaryWriter(config.LOG_DIR + "/real")
step = 0

# Loss + optimizers
loss = nn.BCELoss()
optimizer_gen = torch.optim.Adam(gen.parameters(), lr=config.LR)
optimizer_disc = torch.optim.Adam(disc.parameters(), lr=config.LR)

# Training Process
for epoch in range(config.NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(pokemon_dataloader):
        # Flatten our image
        real = real.view(-1, config.INPUT_SIZE).to(config.DEVICE)
        
        # Discriminator Trainig: max: log(D(real)) + log(1 - D(G(noise)))
        noise = torch.randn(config.BATCH_SIZE, config.Z_DIM).to(config.DEVICE)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        
        lossD_real = loss(disc_real, torch.ones_like(disc_real))
        
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = loss(disc_fake, torch.zeros_like(disc_fake))
        
        lossD = (lossD_real + lossD_fake) / 2
        
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        optimizer_disc.step()
        
        # Generator trainig: min log(1 - D(G(noise))) <-> max: log(D(G(noise)))
        output = disc(fake).view(-1)
        lossG = loss(output, torch.ones_like(output))
        
        gen.zero_grad()
        lossG.backward()
        optimizer_gen.step()
        
        if batch_idx == 0:
            print(
                f"Epoch: [{epoch}/{config.NUM_EPOCHS}] \n",
                f"Loss D: {lossD:.4f}, {lossG:.4f}"
            )
            
            with torch.no_grad():
                fake =  gen(config.FIXED_NOISE).reshape(-1, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
                real_data = real.reshape(-1, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real_data, normalize=True)
                
                writer_fake.add_image(
                    "Pokemon Fake images", img_grid_fake, global_step = step
                )
                
                writer_real.add_image(
                    "Pokemon real images", img_grid_real, global_step = step
                )
                
                step += 1

