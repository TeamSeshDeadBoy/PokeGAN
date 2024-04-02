from torchvision.transforms import transforms
import config
import data
import torch
import utils
import torch.nn as nn
import torchvision
from tqdm import tqdm
from model import Generator, Discriminator, initialize_weights
from torch.utils.tensorboard import SummaryWriter

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

# Instantiate models
gen = Generator(config.Z_DIM, config.IMAGE_CHANNELS, config.FEATURE_GEN).to(config.DEVICE)
disc = Discriminator(config.IMAGE_CHANNELS, config.FEATURE_DISC).to(config.DEVICE)

initialize_weights(disc)
initialize_weights(gen)

# Setting up optimizers and loss
optimizer_gen = torch.optim.Adam(gen.parameters(), lr=config.LR, betas=(0.5, 0.999))
optimizer_disc = torch.optim.Adam(disc.parameters(), lr=config.LR, betas=(0.5, 0.999))

loss = nn.BCELoss()

# tensorboard logging
writer_fake = SummaryWriter(config.LOG_DIR + "/fake")
writer_real = SummaryWriter(config.LOG_DIR + "/real")
step = 0

if config.LOAD_MODEL:
        utils.load_checkpoint(config.CHECKPOINT_GEN, gen, optimizer_gen, config.LR)
        utils.load_checkpoint(config.CHECKPOINT_CRITIC, loss, optimizer_disc, config.LR)
        
gen.train()
disc.train()


# Training Process
for epoch in range(config.NUM_EPOCHS):
    print(f"\n Epoch: [{epoch}/{config.NUM_EPOCHS}]")
    loop = tqdm(pokemon_dataloader, leave=True)
    for batch_idx, (real, _) in enumerate(loop):
        # Flatten our image
        real = real.to(config.DEVICE)
        
        # Discriminator Trainig: max: log(D(real)) + log(1 - D(G(noise)))
        noise = torch.randn((config.BATCH_SIZE, config.Z_DIM, 1, 1)).to(config.DEVICE)
        fake = gen(noise)
        
        disc_real = disc(real).reshape(-1)
        lossD_real = loss(disc_real, torch.ones_like(disc_real))
        
        disc_fake = disc(fake.detach()).reshape(-1)
        lossD_fake = loss(disc_fake, torch.zeros_like(disc_fake))
        
        lossD = (lossD_real + lossD_fake) / 2
        
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        optimizer_disc.step()
        
        # Generator trainig: min log(1 - D(G(noise))) <-> max: log(D(G(noise)))
        output = disc(fake).reshape(-1)
        lossG = loss(output, torch.ones_like(output))
        
        gen.zero_grad()
        lossG.backward()
        optimizer_gen.step()
        
    if config.SAVE_MODEL:
        utils.save_checkpoint(gen, optimizer_gen, filename=config.CHECKPOINT_GEN)
        utils.save_checkpoint(loss, optimizer_disc, filename=config.CHECKPOINT_CRITIC)
            
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
        utils.generate_examples(gen=gen, model_name="DCGAN", n=20)
