# importing libraries
import os
import math
import time
import copy
import imageio.v2 as imageio
import glob
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# hyper parameters
BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 0.0001
EPOCHS = 100
T = 1000
ddim_steps = 50
EMA_DECAY = 0.9999
WARMUP_EPOCHS = 10
SAVE_INTERVAL = 5 
checkpoint_path = 'checkpoint.pth'

class SinusoidalPositionalEmbedding(torch.nn.Module):
    '''
    Discrete scalar timesteps t are mapped into continuous dense vector representations using sinusoidal wave frequencies. 
    This allows the neural network to identify the amount of noise at any given step.
    '''

    def __init__(self, dim):
        super().__init__()
        self.dim = dim # total dimension of output embedding

    def forward(self, time):
        '''
        arguments:
        time: tensor of shape [batch_size,] which contains discrete time steps

        returns: tensor of shape [batch_size, dim] which  contains interleaved sin/cos embeddings.
        '''

        device = time.device
        half_dim = self.dim // 2

        # computing the exponential frequency scale: 10000 ^ (-2i / dim)
        embeddings = math.log(10000) / half_dim
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # computing the outer product [batch_size,1] x [1,half_dim] -> [batch_ize, half_dim]
        embeddings = time[:, None] * embeddings[None, :]

        # concatenate sin and cos frequencies along the channel axis -> [batch_size, dim]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings
    

class ResidualBlock(torch.nn.Module):

    '''
    Residula convolution blocks enhanced with featurewise linear modulation (FiLm). 
    injecting temporal embeddings directly into the latent spatial feature maps
    '''

    def __init__(self, in_channels, out_channels, time_embedding_dim, dropout=0.1):
        super().__init__()

        # MLP matches time embeddings to the target blocks channel depth
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_features=time_embedding_dim, out_features=out_channels),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=out_channels, out_features=out_channels)
        )

        # First Conv block
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.GroupNorm(32, out_channels)

        # Second Conv block
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.GroupNorm(32, out_channels)

        # 1x1 convolution used for matching the input channels to the output channels if the dimensions do not match
        self.residual_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else torch.nn.Identity()

        self.silu = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, time_embedding):
        '''
        arguments:
        x: input feature map [batch_size, in_channels, height, width]
        time_embedding: positional vectors [batch_size, time_embedding_dim]

        returns: processed tensor [batch_size, out_channels, height, width]
        '''

        residual = x

        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)

        # modulate with time embedding
        time_emb = self.time_mlp(time_embedding)
        x = x + time_emb[:, :, None, None]

        x = self.dropout(x)

        # second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.silu(x)

        x = x + self.residual_conv(residual)
        x = self.silu(x)

        return x

class AttentionBlock(torch.nn.Module):
    '''
    self attention module executes over the flattened spatial tokens. this allows the model to long range context across the image grid
    '''

    def __init__(self, channels):
        super().__init__()
        self.norm = torch.nn.GroupNorm(32, channels)

        # multi head attention queries, keys and values within identical feature maps
        self.attention = torch.nn.MultiheadAttention(channels, 4, batch_first=True)

    def forward(self, x):
        '''
        arguments
        x: input tensor [batch_size, channels, height, width]

        returns: tensor [batch_size, channels, height, width]
        '''

        residual = x

        # normalize
        x = self.norm(x)

        # reshape [batch_size, height*width, channels] for attention
        b,c,h,w = x.shape
        x = x.view(b, c, -1).transpose(1,2)

        # apply self attention
        x, _ = self.attention(x,x,x)

        # reshape back
        x = x.transpose(1,2).view(b,c,h,w)

        # add residual connection
        return x + residual

class Unet(torch.nn.Module):

    '''
    U-Net architecture with symmetric Downsampling (Encoder) and Upsampling (Decoder) paths.
    Uses skip connections to preserve localized geometric details contaminated by noise.
    '''

    def __init__(self, in_channels=1, out_channels=1, time_embedding_dim=128, channels=(64,128,256,512), dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding_dim = time_embedding_dim
        self.channels = channels

        # embedding extractor for noise step scheduling
        self.time_embedding = SinusoidalPositionalEmbedding(time_embedding_dim)
        self.initial_conv = torch.nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        # Encoder
        self.downsample_blocks = torch.nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downsample_blocks.append(torch.nn.ModuleDict({
                "res1": ResidualBlock(channels[i], channels[i], time_embedding_dim, dropout),
                "res2": ResidualBlock(channels[i], channels[i], time_embedding_dim, dropout),
                "attn": AttentionBlock(channels[i]),
                "down": torch.nn.Conv2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1),
            }))

        # Bottleneck
        self.middle = torch.nn.ModuleList([
            ResidualBlock(channels[-1], channels[-1], time_embedding_dim, dropout),
            AttentionBlock(channels[-1]),
            ResidualBlock(channels[-1], channels[-1], time_embedding_dim, dropout),
        ])

        # Decoder
        self.upsample_blocks = torch.nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.upsample_blocks.append(torch.nn.ModuleDict({
                "up": torch.nn.ConvTranspose2d(channels[i], channels[i-1], kernel_size=4, stride=2, padding=1),
                # Note: Channel count doubles during input due to concatenated skip links
                "res_main": ResidualBlock(channels[i-1] * 2, channels[i-1], time_embedding_dim, dropout),
                "res2": ResidualBlock(channels[i-1], channels[i-1], time_embedding_dim, dropout),
                "attn": AttentionBlock(channels[i-1]),
            }))

        # Output
        self.final_res1 = ResidualBlock(channels[0] * 2, channels[0], time_embedding_dim, dropout)
        self.final_res2 = ResidualBlock(channels[0], channels[0], time_embedding_dim, dropout)
        self.final_norm = torch.nn.GroupNorm(32, channels[0])
        self.final_conv = torch.nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)

    def forward(self, x, time):

        # transform scalar indices into positional vector features
        time_emb = self.time_embedding(time)
        
        # extract spatial entry maps and stage first skip entry
        x = self.initial_conv(x)
        skips = [x] # Store initial conv output

        # Encoder
        for block in self.downsample_blocks:
            x = block["res1"](x, time_emb)
            x = block["res2"](x, time_emb)
            x = block["attn"](x)
            skips.append(x) # Store features BEFORE downsampling 
            x = block["down"](x)

        # Bottleneck
        x = self.middle[0](x, time_emb)
        x = self.middle[1](x)
        x = self.middle[2](x, time_emb)

        # Decoder
        for block in self.upsample_blocks:
            skip = skips.pop() # Get the last saved feature map
            x = block["up"](x)
            
            # spatial alignment if scale dimension does not match
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = block["res_main"](x, time_emb)
            x = block["res2"](x, time_emb)
            x = block["attn"](x)

        # Final connection with initial input convolution map
        skip = skips.pop()
        x = torch.cat([x, skip], dim=1)
        x = self.final_res1(x, time_emb)
        x = self.final_res2(x, time_emb)
        x = self.final_norm(x)
        x = torch.nn.functional.silu(x)
        x = self.final_conv(x) # maps features to the final image channels

        return x

class EMA:
    '''
    Maintains a shadow copy of the model weights smoothed via Exponential Moving Average.
    '''
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow_model = copy.deepcopy(model)
        self.shadow_model.eval()
        for param in self.shadow_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for shadow_param, model_param in zip(self.shadow_model.parameters(), model.parameters()):
            shadow_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow_model.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow_model.load_state_dict(state_dict)


def variance_scheduler(T, s=0.008):

    '''
    Implements a cosine noise variance scheduler. This prevents sudden noise jumps which makes the optimization smoother.
    '''

    steps = T + 1
    x = torch.linspace(0, T, steps)

    # calculate alpha cumprod values based on cosine trajectory
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    # Calculate step specific betas from alphas_cumprod
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clamp(betas, 0, 0.999)
    
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0) # Match indexing to length T [0 to T-1]
    
    return alphas, alphas_cumprod, betas


def forward_process(x, alpha_cumprod, device):

    '''
    executes closed form diffusion forward pass. It directly samples noisy image at step t without running intermediate iterations.
    '''

    batch_size = x.shape[0]

    # randomly selects a diffusion timestep uniform spectrum [0, T-1] per sample
    t = torch.randint(low=0, high=T, size=(batch_size,), dtype=torch.int64, device=device)
    alpha_cm = alpha_cumprod[t]

    # generate gaussian white noise
    noise = torch.randn_like(x)
    
    # reparameterization trick calculation
    x_t = torch.sqrt(alpha_cm).view(-1,1,1,1) * x + torch.sqrt(1-alpha_cm).view(-1,1,1,1) * noise
    return x_t, noise, t

def reverse_process(model, batch_size, T, alphas, alphas_cumprod, betas, device):

    '''
    standard ddpm markovian step generative sampling pipeline
    iteratively runs backward from pure gaussian white noise to clean samples step by step
    '''
    
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28, device=device) # start with pure noise

    with torch.no_grad():
        for i in range(T - 1, -1, -1): # go backwards from T-1 down to 0
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            alpha_t = alphas[i]
            alpha_cumprod_t = alphas_cumprod[i]
            beta_t = betas[i]
            
            if i > 0:
                alpha_cumprod_prev = alphas_cumprod[i-1]

                # calculate the true posterior variance parameter
                variance = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * beta_t
                noise = torch.randn_like(x)
            else:
                variance = torch.tensor(0.0, device=device)
                noise = torch.tensor(0.0, device=device) # final step adds no variance

            # predict structural noise using unet estimation
            pred_noise = model(x, t)
            
            # reverse execution mean calculation step
            mean = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise
            )
            
            # step update
            x = mean + torch.sqrt(variance) * noise


    return x


def reverse_process_ddim(model, batch_size, T, alphas_cumprod, device, ddim_steps=50, eta=0.0):
    '''
    DDIM, non markovian acceleration sampler which allows jumping accross sub sampled time increments to sample significantly faster
    eta controls the determinism. 0 corresponds to deterministic ddim and 1 corresponds to ddpm
    '''

    model.eval()
    x = torch.randn(batch_size, 1, 28, 28, device=device)
    
    # subsample step grid stride to skip chunks of time
    # times = torch.linspace(0, T - 1, ddim_steps, dtype=torch.long, device=device)
    times = torch.arange(0, T, T // ddim_steps, device=device)
    times = times[-ddim_steps:]
    times_prev = torch.cat([torch.tensor([-1], device=device), times[:-1]])
    
    # We iterate backwards through our subsequence
    with torch.no_grad():
        for i in range(ddim_steps - 1, -1, -1):
            t = torch.full((batch_size,), times[i], device=device, dtype=torch.long)
            t_prev = torch.full((batch_size,), times_prev[i], device=device, dtype=torch.long)
            
            # Get alpha_cumprod for current and previous step
            alpha_cm_t = alphas_cumprod[times[i]]
            alpha_cm_prev = alphas_cumprod[times_prev[i]] if times_prev[i] >= 0 else torch.tensor(1.0, device=device)
            
            # model evaluates current noisy state
            pred_noise = model(x, t)
            
            # estimate original data clean start state
            pred_x0 = (x - torch.sqrt(1 - alpha_cm_t) * pred_noise) / torch.sqrt(alpha_cm_t)
            
            # derive stochastic noise multiplier coefficient sigma
            sigma_t = eta * torch.sqrt((1 - alpha_cm_prev) / (1 - alpha_cm_t) * (1 - alpha_cm_t / alpha_cm_prev))
            
            # form direction vector pointing back current time step state
            pred_dir_xt = torch.sqrt(1 - alpha_cm_prev - sigma_t**2) * pred_noise
            
            # compute overall jump state updates
            x = torch.sqrt(alpha_cm_prev) * pred_x0 + pred_dir_xt
            
            if sigma_t > 0:
                x += sigma_t * torch.randn_like(x)

    return x

def get_lr_scheduler(optimizer, total_epochs, warmup_epochs):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Linear warmup
            return float(current_epoch) / float(max(1, warmup_epochs))
        
        # Cosine annealing
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(train_loader, val_loader, model, ema, criterion, optimizer, alpha_cumprod, T, device):

    # training phase
    model.train()
    train_loss = 0.0

    for x, _ in train_loader:
        x = x.to(device)

        # draw a single step random forward noise calculation pass
        x_t, noise, t = forward_process(x, alpha_cumprod, device)

        # estimate random noise vector added to the original image
        predicted_noise = model(x_t, t)

        # compute mse loss against true target noise
        loss = criterion(predicted_noise, noise)

        # back propagation and optimization pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # validation phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)

            # forward process
            x_t, noise, t = forward_process(x, alpha_cumprod, device)

            # model prediction
            predicted_noise = model(x_t, t)

            # compute loss
            loss = criterion(predicted_noise, noise)
            val_loss += loss.item()

        val_loss /= len(val_loader)

        return train_loss, val_loss


def generate_images(model, batch_size, T, alpha_cumprod, device, ddim_steps=50, num_batches=1):

    '''
    wrapper pipeline function initiating accelerated inference generations using DDIM routine loops.
    '''

    model.eval()
    all_images = []
    
    with torch.no_grad():
        for _ in range(num_batches):
            # Notice we pass ddim_steps here
            images = reverse_process_ddim(model, batch_size, T, alpha_cumprod, device, ddim_steps=ddim_steps)
            all_images.append(images)
    
    return torch.cat(all_images, dim=0)


def plot_generated_images(images, num_images=16, save_path='generated_image.png'):

    '''
    Inverts normalizations, handles safe tensor clamping, and compiles images into a structured grid.
    '''
    
    # unnormalize images (reverse the normalization applied in transform)
    images = images * 0.5 + 0.5  # reverse normalization: x = (x * std) + mean
    images = torch.clamp(images, 0, 1)  # clamp to [0, 1]
    
    # create grid
    num_images = min(num_images, images.shape[0])
    grid_size = int(math.sqrt(num_images))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    axes = axes.flatten()
    
    for i in range(num_images):
        img = images[i].cpu().squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Generated images saved to {save_path}.')
    plt.close()


def create_training_gif(source_dir, gif_name='training_progress.gif'):
    images = []
    # Get all png files and sort them numerically by epoch
    file_list = sorted(glob.glob(os.path.join(source_dir, 'epoch_*.png')))
    
    for filename in file_list:
        images.append(imageio.imread(filename))
    
    if images:
        # duration is in seconds per frame
        imageio.mimsave(gif_name, images, fps=5) 
        print(f"Training GIF saved as {gif_name}")
    else:
        print("No images found to create GIF.")


def save_checkpoint(epoch, model, ema, optimizer, scheduler, file_path):
    
    # saving the model while training
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }

    torch.save(checkpoint, file_path)
    print(f"Checkpoint saved at epoch: {epoch}.")


def load_checkpoint(model, ema, optimizer, scheduler, filepath, device):
    
    # if checkpoint exists
    try:
        checkpoint = torch.load(filepath, map_location=device)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        print(f'Checkpoint loaded. Resuming training from epoch: {start_epoch}.')
        return start_epoch

    # if checkpoint does not exist
    except FileNotFoundError:
        print('No checkpoint found. Training from scratch.')
        return 0

def main():
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    print('Device used:', device)
    print('Verion of torch:', torch.__version__)

    # transform
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

    # getting the dataset
    train_dataset_full = torchvision.datasets.FashionMNIST(root='./data', download=True, train=True, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset_full, [54000, 6000])

    # displaying the number of examples in the training and validation sets
    print('Number of examples in the training set', len(train_dataset))
    print('Number of examples in the validation set', len(val_dataset))

    # creating the dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # checking the shape of a batch of images and labels
    for x, y in train_loader:
        print('Shape of x', x.shape)
        print('Shape of y', y.shape)
        break
    
    # initializing the scheduler
    alpha, alpha_cumprod, beta  = variance_scheduler(T) # variance scheduler
    alpha, alpha_cumprod, beta  = alpha.to(device), alpha_cumprod.to(device), beta.to(device)

    # creating the model
    model = Unet().to(device)
    ema = EMA(model, decay=EMA_DECAY)

    # optimizer and criterion
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=LR)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scheduler = get_lr_scheduler(optimizer, EPOCHS, WARMUP_EPOCHS)

    # create a directory if it does not exist for saving the generated images
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir='runs/fashion_mnist_diffusion') # creating a folder for tensorboard logs

    start_epoch = load_checkpoint(model, ema, optimizer, scheduler, checkpoint_path, device)

    print('\nTraining the model...')
    for epoch in range(start_epoch, EPOCHS):

        start_time = time.time()
        train_loss, val_loss = train_step(train_loader, val_loader, model, ema, criterion, optimizer, alpha_cumprod, T, device)
        end_time = time.time()

        current_lr = optimizer.param_groups[0]['lr'] # current lr

        print(f'Epoch:{epoch+1:03d} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train time: {end_time-start_time:.2f} sec')

        # tensorboard logging
        writer.add_scalar('Loss/Train', train_loss, epoch + 1)
        writer.add_scalar('Loss/Validation', val_loss, epoch + 1)
        writer.add_scalar('Learning_rate', current_lr, epoch + 1)

        # for saving images every n epoch while training the model
        if (epoch + 1) % SAVE_INTERVAL == 0:
            sample_images = generate_images(ema.shadow_model, batch_size=16, T=T, alpha_cumprod=alpha_cumprod, device=device, ddim_steps=ddim_steps)
            save_path = os.path.join(output_dir, f'epoch_{epoch+1:03d}.png')
            plot_generated_images(sample_images, num_images=16, save_path=save_path)
            
        scheduler.step() # updating the scheduler

        save_checkpoint(epoch+1, model, ema, optimizer, scheduler, checkpoint_path)

    print('Training completed. Creating a gif from the generated images')
    create_training_gif(output_dir)
    writer.close()


if __name__ == '__main__':
    main()