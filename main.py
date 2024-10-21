# Imports
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from src.dcgan import Generator, Discriminator
from src.data_utils import prepare_dataloader
from src.vis_utils import setup_directories, save_results, visualise_dcgan_training, visualise_dcgan_test
from src.train import train_network, generate_test_images

# Set constants
IMAGE_SIZE = 64
IMAGE_GEN_BATCH_SIZE = 64 # Matches default batch size for convenience
N_CHANNELS = 3 # Number of channels in the training and output image (i.e. 3 for colour images)
Z_DIM = 100 # Size of latent vector Z, i.e. generator input (sourced from paper)
NGF = IMAGE_SIZE # Size of feature maps in generator (sourced from paper)
NDF = IMAGE_SIZE # Size of feature maps in discriminator (sourced from paper
LEARNING_RATE = 2e-4 # Learning rate for optimisers (sourced from paper)
BETA1 = 0.5 # Beta1 hyperparameter for Adam optimisers (sourced from paper)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = 'data/impressionist_landscapes'

# Default values for command-line arguments
DEFAULT_DATASET = 'full'
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 2
DEFAULT_NUM_EPOCHS = 100

# Parse command-line arguments
def get_parser():
    parser = argparse.ArgumentParser(description='Training a Deep Convolutional Generative Adversarial Network on Impressionism artworks in Pytorch.')
    
    parser.add_argument('--mode', choices=['train', 'test'], required=True,
                        help='Choose the mode: training the model (train), or generate images with the trained model on random noise samples (test).')
    parser.add_argument('--checkpoint_path', type=str, default=None, 
                        help='Path to the checkpoint file to load the model state. If not provided, the weights will be randomly initialised.')
    parser.add_argument('--dataset', choices=['full', 'mini'], default=DEFAULT_DATASET, help='Dataset option: full or mini.')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for training.')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS, help='Number of data loading workers.')
    parser.add_argument('--num_epochs', type=int, default=DEFAULT_NUM_EPOCHS, help='Number of epochs for training.')
    parser.add_argument('--model_name', type=str, default=None, help='Optional name for the model to be run. If provided, checkpoints and results will be saved under checkpoint/model_name/ and results/model_name/.')
    
    return parser

def main():
    # Retrieve variables from arguments
    parser = get_parser()
    args = parser.parse_args()

    # Prepare dataset       
    dataset, dataloader = prepare_dataloader(
        data_dir=DATA_DIR, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        dataset_type=args.dataset
    )

    # Initialise generator and discriminator networks
    discriminator = Discriminator(
        ndf=NDF, 
        n_channels=N_CHANNELS,
        device=DEVICE,
        checkpoint_path=args.checkpoint_path
    )
    
    generator = Generator(
        z_dim=Z_DIM, 
        ngf=NGF, 
        n_channels=N_CHANNELS,
        device=DEVICE,
        checkpoint_path=args.checkpoint_path
    )
    
    # Set loss function and parameters
    criterion = nn.BCELoss()
    optimiser_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimiser_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    # Create batch of latent vectors to feed into generator
    fixed_noise = torch.randn(IMAGE_GEN_BATCH_SIZE, Z_DIM, 1, 1, device=DEVICE)
    
    # Setup directory
    checkpoint_dir, results_dir, images_dir = setup_directories(
        run_mode=args.mode,
        model_name=args.model_name, 
        checkpoint_base='checkpoint', 
        results_base='results',
        images_base = 'images'
    )

    if args.mode == 'train':
        # Train model and save checkpoints
        loss_history_iter, loss_history_epoch, epoch_image_history = train_network(
            discriminator=discriminator, 
            generator=generator, 
            dataloader=dataloader, 
            num_epochs=args.num_epochs, 
            criterion=criterion, 
            optimiser_d=optimiser_d, 
            optimiser_g=optimiser_g, 
            fixed_noise=fixed_noise,
            z_dim=Z_DIM,
            device=DEVICE, 
            checkpoint_dir=checkpoint_dir
        )
    
        # Save results
        save_results(
            loss_history_iter=loss_history_iter, 
            loss_history_epoch=loss_history_epoch, 
            save_dir=results_dir
        )

        # Save generated images from training DCGAN
        visualise_dcgan_training(
            epoch_image_history=epoch_image_history, 
            save_dir=images_dir
        )
    
    else:
        # Generate new images from trained DCGAN and save
        fake_images = generate_test_images(
            generator=generator, 
            fixed_noise=fixed_noise
        )
        
        visualise_dcgan_test(
            fake_images=fake_images, 
            save_dir=images_dir
        )

if __name__ == "__main__":
    main()