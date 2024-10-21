# Imports
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def setup_directories(model_name=None, run_mode='train', checkpoint_base='checkpoint', results_base='results', images_base = 'images'):
    """Setup directories for saving checkpoints, results, and output images."""
    checkpoint_dir = f'{checkpoint_base}/{model_name}' if model_name else 'checkpoint'
    results_dir = f'{results_base}/{model_name}' if model_name else 'results'
    images_dir = f'{images_base}/{model_name}' if model_name else 'images'

    if run_mode == 'train':
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    return checkpoint_dir, results_dir, images_dir

def plot_image(image):
    """Plot an image after denormalising it."""
    # Denormalise image
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    image = image.permute(1,2,0) * std + mean

    # Plot image
    plt.imshow(image)

def visualise_dataset(dataset, save_dir):
    """Create a 8x8 grid of random images from dataset and save output."""
    # Filepath for saving
    path_image = os.path.join(save_dir, f'dataset_vis.png')

    # Randomly select and plot 64 images 
    indices = np.arange(len(dataset))
    rand_indices = np.random.choice(indices, size=min(64, len(indices)), replace=False)
    
    plt.figure(figsize=(8, 8))
    for i, idx in enumerate(rand_indices):
        plt.subplot(8, 8, i + 1)
        plot_image(dataset[idx])
        plt.axis('off')
        
    plt.tight_layout(pad=0.2) 
    plt.savefig(path_image)
    plt.show()

def save_history_and_plots(save_dir, time_scale, loss_history):
    """Save loss history, and discriminator and generator loss plots."""
    if time_scale not in ['epoch', 'iteration']:
        raise ValueError("Invalid time_scale value. Expected 'epoch' or 'iteration'.")
    
    # Filepaths
    path_loss_history = os.path.join(save_dir, f'loss_history_{time_scale}.txt')
    path_loss_plot = os.path.join(save_dir, f'plot_loss_vs_{time_scale}.png')
    
    # Save loss
    with open(path_loss_history, 'w') as f:
        for t, loss_d, loss_g in loss_history:
            f.write(f'{t}\t{loss_d:.4f}\t{loss_g:.4f}\n')
    
    # Extract t, discriminator loss, and generator loss
    t, loss_D, loss_G = zip(*loss_history)

    # Plot and save loss_d and loss_g vs t
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Training Loss")
    plt.plot(loss_G,label="loss_G")
    plt.plot(loss_D,label="loss_D")
    plt.xlabel(f'{time_scale.capitalize()}s')
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(path_loss_plot)
    plt.show()

def save_results(loss_history_iter, loss_history_epoch, save_dir):
    """Helper function to save both iteration-level and epoch-level loss histories and generate corresponding plots."""
    save_history_and_plots(save_dir, 'iteration', loss_history_iter)
    save_history_and_plots(save_dir, 'epoch', loss_history_epoch)

def visualise_fake_images(fake_images, path_image, show_image=False):
    """Create a 8x8 grid of fake images and save output."""   
    plt.figure(figsize=(8, 8))
    for i, img in enumerate(fake_images):
        plt.subplot(8, 8, i + 1)
        plot_image(img)
        plt.axis('off')
        
    plt.tight_layout(pad=0.3) 
    plt.savefig(path_image)
    if show_image:
        plt.show()

def visualise_dcgan_training(epoch_image_history, save_dir):
    """Visualise and save all generated images from DCGAN during training."""
    with torch.no_grad():
        for epoch, fake_images in epoch_image_history:
            path_image = os.path.join(save_dir, f'dcgan_output_epoch_{epoch}.png') 
            visualise_fake_images(fake_images, path_image)

def visualise_dcgan_test(fake_images, save_dir):
    """Visualise and save generated images from trained DCGAN at test-time."""
    path_image = os.path.join(save_dir, f'dcgan_generated_images.png')
    visualise_fake_images(fake_images, path_image, show_image=True)