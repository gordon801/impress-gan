# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import os

def weights_init(m):
    """
    Randomly initialise model weights from a normal distribution with mean=0 and std=0.02 (based on DCGAN paper). 
    BatchNorm initialised with gamma ~1 and beta = 0.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """
     Generator model implementation according to the Deep Convolutional Generative Adversarial Network (DCGAN) paper.
    """
    def __init__(self, z_dim, ngf, n_channels, device, checkpoint_path=None):
        """Initialise the Generator model.
        
        Args:
            z_dim (int): The dimensionality of the input noise vector (latent space).
            ngf (int): The depth of the feature maps that are propagated through the generator.
            n_channels (int): The number of channels in the input image (3 for RGB images).
            device (torch.device): The device on which the model should run (e.g., "cuda" or "cpu").
            checkpoint_path (str, optional): The path to the checkpoint file to load weights. Defaults to None.
        """
        super(Generator, self).__init__()

        # Define Generator architecture
        self.model = nn.Sequential(
            # Input: z_dimx1x1, Output: (ngf*8)x4x4
            nn.ConvTranspose2d(z_dim, ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            
            # Input: (ngf*8)x4x4, Output: (ngf*4)x8x8
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            
            # Input: (ngf*4)x8x8, Output: (ngf*2)x8x8
            nn.ConvTranspose2d(ngf*4, ngf*2,  kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            
            # Input: (ngf*2)x16x16, Output: ngfx32x32
            nn.ConvTranspose2d(ngf*2, ngf,  kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            # Input: ngfx32x32, Output: 3x64x64
            nn.ConvTranspose2d(ngf, n_channels,  kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        # Load weights from checkpoint if provided, else randomly initialise weights
        if checkpoint_path:
            self._load_checkpoint_weights(checkpoint_path)
        else:
            self.model.apply(weights_init)
        
        # Move model to the specified device
        self.model = self.model.to(device)

    def _load_checkpoint_weights(self, checkpoint_path):
        """Load the model weights from the specified checkpoint path."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint['generator_state_dict']
            self.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"No checkpoint file found at {checkpoint_path}.")
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

class Discriminator(nn.Module):
    """
    Discriminator model implementation according to the DCGAN paper.
    """
    def __init__(self, ndf, n_channels, device, checkpoint_path=None):
        """Initialise the Discriminator model.
        
        Args:
            ndf (int): The depth of the feature maps that are propagated through the discriminator.
            n_channels (int): The number of channels in the input image (3 for RGB images).
            device (torch.device): The device on which the model should run (e.g., "cuda" or "cpu").
            checkpoint_path (str, optional): The path to the checkpoint file to load weights. Defaults to None.
        """
        super(Discriminator, self).__init__()
        
        # Define Discriminator architecture
        self.model = nn.Sequential(
            # Input: 3x64x64, Output: ndfx32x32
            nn.Conv2d(n_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Input: ndfx32x32, Output: (ndf*2)x16x16
            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Input: (ndf*2)x16x16, Output: (ndf*4)x8x8
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Input: (ndf*4)x8x8, Output: (ndf*8)x4x4
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Input: (ndf*8)x4x4, Output: 1x1x1
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        # Load weights from checkpoint if provided, else randomly initialise weights
        if checkpoint_path:
            self._load_checkpoint_weights(checkpoint_path)
        else:
            self.model.apply(weights_init)
        
        # Move model to the specified device
        self.model = self.model.to(device)

    def _load_checkpoint_weights(self, checkpoint_path):
        """Load the model weights from the specified checkpoint path."""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint['discriminator_state_dict']
            self.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"No checkpoint file found at {checkpoint_path}.")
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)