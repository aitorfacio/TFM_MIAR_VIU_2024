#%matplotlib inline
import argparse
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import argparse
import time
import random
from string import ascii_letters, digits
from tqdm import tqdm
# Root directory for dataset

def create_next_subfolder(path):
    created_path = Path(path)
    created_path.mkdir(exist_ok=True, parents=True)
    chk_existing_folders = [int(p.name) for p in created_path.iterdir()]
    chk_next_folder_num = max(chk_existing_folders) + 1 if chk_existing_folders else 1
    chk_current_session_dir = created_path / str(chk_next_folder_num)
    chk_current_session_dir.mkdir(parents=True, exist_ok=True)
    created_path = chk_current_session_dir
    return created_path



# Create ArgumentParser object

# Parse the arguments

# Assign parsed arguments to variables


## Plot some training images
#real_batch = next(iter(dataloader))
#plt.figure(figsize=(8,8))
#plt.axis("off")
#plt.title("Training Images")
#plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
#plt.show()



# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)



class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, ndf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def generate(opt):
    nc = opt.nc
    nz = opt.nz
    ngf = opt.ngf
    ndf = opt.ndf
    ngpu = opt.ngpu
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    netG = Generator(ngpu, nz=nz, ngf=ngf, nc=nc).to(device)
    netG.load_state_dict(torch.load(opt.generator_path))
    netG.cuda()
    netG.eval()

    remaining_images = opt.number_of_generated_images
    dirs = []
    while remaining_images > 0:
        if opt.images_per_identity > 1:
            base_z = torch.randn(1, nz, 1, 1).cuda()
            variations = torch.randn(opt.images_per_identity, nz, 1, 1, device='cuda') * 0.8
            noise = base_z + variations
        else:
            noise = torch.randn(remaining_images, nz, 1, 1).cuda()
            remaining_images = 0
        imgs = netG(noise)
        if opt.output:
            current_session_dir = Path(opt.output)
            current_session_dir.mkdir(parents=True, exist_ok=True)
        else:
            images_target_dir = Path("images") / "my_gan" / f"{opt.mode}"
            images_target_dir.mkdir(parents=True, exist_ok=True)
            current_session_dir = create_next_subfolder(images_target_dir)

        while True:
            identity = ''.join(random.choice(ascii_letters + digits) for _ in range(8))
            if identity not in dirs:
                break
        image_path = current_session_dir / identity
        image_path.mkdir(parents=True)
        dirs.append(identity)
        for i, img in tqdm(enumerate(imgs)):
            save_path = image_path / f"generated_img_{i}.png"
            vutils.save_image(img, save_path)
        remaining_images -= opt.images_per_identity


def train(opt):
    epochs_without_improvement = 0
    if opt.output:
        current_session_dir = Path(opt.output) / 'images'
        current_session_dir.mkdir(parents=True, exist_ok=True)
    else:
        images_target_dir = Path("images") / "my_gan" / f"{opt.mode}"
        images_target_dir.mkdir(parents=True, exist_ok=True)
        current_session_dir = create_next_subfolder(images_target_dir)
    dataroot = opt.dataroot
    workers = opt.workers
    batch_size = opt.batch_size
    image_size = opt.image_size
    nc = opt.nc
    nz = opt.nz
    ngf = opt.ngf
    ndf = opt.ndf
    num_epochs = opt.num_epochs
    lr = opt.lr
    beta1 = opt.beta1
    ngpu = opt.ngpu
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # Set random seed for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)  # Needed for reproducible results
    # Create the generator
    netG = Generator(ngpu, nz=nz, ngf=ngf, nc=nc).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    # Create the Discriminator
    netD = Discriminator(ngpu, nc=nc, ndf=ndf).to(device)

    # Handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(weights_init)

    # Print the model
    print(netD)
    # We can use an image img dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    if opt.output:
        checkpoint_path = Path(opt.output) / 'checkpoints'
        checkpoint_path.mkdir(parents=True, exist_ok=True)
    else:
        checkpoint_path = create_next_subfolder('checkpoints')
    with open(checkpoint_path/ "hyperparamenters.txt", 'w') as f:
        f.write(str(vars(opt)))
    # Inside your training loop
    losses_file_path = checkpoint_path / "training_losses.txt"
    with open(losses_file_path, "w") as file:
        file.write("Epoch,Loss_D,Loss_G,D(x),D(G(z))\n")  # Header for the CSV file
    # Initialize the ``BCELoss`` function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    lowest_G_loss = math.inf
    lowest_DGZ = math.inf

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                with open(losses_file_path, "a") as file:
                    file.write(f"{epoch},{i},{errD.item()},{errG.item()}, {D_x}, {D_G_z1}\n")
                image_path = current_session_dir / f"generated_image_iter_{iters}.png"
                vutils.save_image(fake, image_path, padding=2, normalize=True)

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        if G_losses[-1] < lowest_G_loss:
            lowest_G_loss = G_losses[-1]
            generator_checkpoint_path = checkpoint_path / f'generator_best.pth'
            discriminator_checkpoint_path = checkpoint_path / f'discriminator_best.pth'
            torch.save(netG.state_dict(), str(generator_checkpoint_path))
            torch.save(netD.state_dict(), str(discriminator_checkpoint_path))
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= opt.patience:
            print("Stop training due to no improvement.")
            break

    generator_checkpoint_path = checkpoint_path / f'generator_final.pth'
    discriminator_checkpoint_path = checkpoint_path / f'discriminator_final.pth'
    torch.save(netG.state_dict(), str(generator_checkpoint_path))
    torch.save(netD.state_dict(), str(discriminator_checkpoint_path))
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(str(checkpoint_path / 'training_losses_final.png'))
    plt.close()

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    writergif = animation.PillowWriter(fps=30)

    ani.save("generated_animation.gif", writer=writergif)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCGAN Training')

    # Add arguments
    parser.add_argument('--dataroot', type=str,
                        default=r"C:\Users\Aitor\datasets\race_unbalance\African_112x112_single\African_112x112",
                        help='Root directory of the dataset')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training')
    parser.add_argument('--image_size', type=int, default=64, help='Spatial size of training images. All images will be resized to this size using a transformer.')
    parser.add_argument('--nc', type=int, default=3, help='Number of channels in the training images. For color images, this is 3')
    parser.add_argument('--nz', type=int, default=100, help='Size of z latent vector (i.e. size of generator input)')
    parser.add_argument('--ngf', type=int, default=64, help='Size of feature maps in generator')
    parser.add_argument('--ndf', type=int, default=64, help='Size of feature maps in discriminator')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate for optimizers')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 hyperparameter for Adam optimizers')
    parser.add_argument('--ngpu', type=int, default=1, help='Number of GPUs available. Use 0 for CPU mode.')
    parser.add_argument("--mode", type=str, choices=['train', 'generate'], default='train',
                        help="Mode of operation: train or generate")
    parser.add_argument('--patience', type=int, default=3, help='Number of epochs without improvement to wait.')
    parser.add_argument("--generator_path", type=str, default="generator.pth", help="Path to the generator model file")
    parser.add_argument("--output", type=str, help="Exact path to save the results to." )
    parser.add_argument("--number_of_generated_images", type=int, default=10)
    parser.add_argument("--images_per_identity", type=int, default=10, help="Path for the dataset.")
    opt = parser.parse_args()
    if opt.mode == "train":
        train(opt)
    elif opt.mode == "generate":
        generate(opt)
