# Imports
import os
import torch
from flask import Flask, render_template, send_file
from main import DEVICE, Z_DIM, NGF, N_CHANNELS, IMAGE_GEN_BATCH_SIZE
from src.dcgan import Generator
from src.train import generate_test_images
from src.vis_utils import visualise_fake_images

# Set constants
MODEL_PATH = 'checkpoint/gl-epoch150/checkpoint_epoch_130.pth.tar' # Replace with appropriate checkpoint file path
SAVE_DIR = 'images/app'

app = Flask(__name__)

# Ensure directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialise generator
generator = Generator(
    z_dim=Z_DIM, 
    ngf=NGF, 
    n_channels=N_CHANNELS,
    device=DEVICE,
    checkpoint_path=MODEL_PATH
)

def uniquify_filename(save_dir):
    """Generate a unique filename by appending a number as required."""
    i = 0
    while True:
        if i > 0:
            filename = f'impressgan_images_{i}.png'
        else:
            filename = f'impressgan_images.png'

        if not os.path.exists(os.path.join(save_dir, filename)):
            return filename
        i += 1

def generate_images():
    """Generates a grid of fake images, and saves it with a unique filename. Returns the path to the saved image."""
    fixed_noise = torch.randn(IMAGE_GEN_BATCH_SIZE, Z_DIM, 1, 1, device=DEVICE)

    fake_images = generate_test_images(
        generator=generator, 
        fixed_noise=fixed_noise
    )
    
    filename = uniquify_filename(save_dir=SAVE_DIR)
    path_image = os.path.join(SAVE_DIR, filename)

    visualise_fake_images(
        fake_images=fake_images, 
        path_image=path_image
    )

    return path_image

@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """Generate images and return the generated image."""
    image_path = generate_images()
    return send_file(image_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)