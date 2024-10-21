![impressgan-demo](https://github.com/user-attachments/assets/a35c150b-b2cf-4c15-957c-1df9d000baeb)

# ImpressGAN: AI-Generated Impressionist Art

This project involves training a Deep Convolutional Generative Adversarial Network (DCGAN) on a dataset of Impressionist artworks to generate new, creative pieces. This is an implementation of the research paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434) by Alec Radford, Luke Metz, and Soumith Chintala (ICLR 2016).

This repository contains two key components:
1. A [Jupyter notebook](https://github.com/gordon801/impress-gan/blob/main/impress-gan.ipynb) that provides a demo of the training and image generation processes.
2. A [Flask web application](https://github.com/gordon801/impress-gan/blob/main/app.py) that deploys and serves the trained DCGAN model, allowing users to generate new Impressionist-style images through a simple web interface.

## Architecture
![image](https://github.com/user-attachments/assets/0f6d324e-68d0-4c02-b709-0337b6c3c301)

The Deep Convolutional Generative Adversarial Network (DCGAN) consists of a Generator network and a Discriminator network, which are trained adversarially within a convolutional neural network (CNN) framework to generate realistic images. We trained the network with Binary Cross-Entropy loss and assessed the quality of its generated images visually.

Architecture guidelines (from DCGAN paper):
- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.

## DCGAN Training Progression
![impressgan-training-demo-resized](https://github.com/user-attachments/assets/66dcca63-80c8-4991-8913-0c1ce0df320d)

This is an animation that visualises the GAN's training progress over time. It showcases the evolution of the generated images across different epochs, and how the model gradually learns to produce more realistic images resembling the Impressionist artwork dataset.

## Example Outputs
![dcgan_generated_images](https://github.com/user-attachments/assets/ddbad598-f04f-440b-b299-ec69772b0a76)
![impressgan_images_6](https://github.com/user-attachments/assets/ac8da2c4-f4fb-4454-b682-c5341dfa91e7)


## Project Structure
```
impress-gan/
├── data/
│   └── impressionist_landscapes/
├── src/
│   ├── data_utils.py
│   ├── dcgan.py
│   ├── train.py
│   └── vis_utils.py
├── templates/
│   └── index.html
├── app.py
├── main.py
└── ...
```
### Data
This model was trained and tested on the `Impressionist-landscapes-paintings` Kaggle dataset, which consists of 500 Impresionist Landscape Paintings in 1024x1024 RGB format. The dataset can be downloaded from [here](https://www.kaggle.com/datasets/robgonsalves/impressionistlandscapespaintings) and should be saved as `impressionist_landscapes` in the `data` folder in the root-level directory. 

### Scripts
- `src/data_utils.py`: Provides functions for loading and transforming the Impressionism Artwork dataset.
- `src/dcgan.py`: Implements a DCGAN with custom Generator and Discriminator models, including functionalities for weight initialisation, and loading from checkpoints.
- `src/train.py`: Implements functions for training a DCGAN, including functionalities to track loss values, save model checkpoints, and generate images from fixed noise. It also provides a method for generating test images using the trained generator.
- `src/vis_utils.py`: Provides utility functions for organising directories, saving training histories, generating plots, and visualising images from the dataset and from the DCGAN's generated outputs.
- `main.py`: Serves as the entry point for the project, managing command-line arguments and orchestrating the initialisation, training, and testing of a DCGAN model on Impressionist artworks.
- `app.py`: A Flask application that serves a trained DCGAN model to generate new images based on images of Impressionist artwork.

### Running `main.py` and `app.py`
To train the model on the full dataset for 150 epochs, run:
```
python main.py --mode train --num_epochs 150 --model_name my_model
```
To generate new images using your trained model at epoch 130, run:
```
python main.py --mode test --checkpoint_path checkpoint/my_model/checkpoint_epoch_130.pth.tar --model_name my_model
```
To deploy your trained model to a web application, run:
```
python app.py
```

### Steps to Reproduce
1. Clone this repository:
```
git clone https://github.com/gordon801/impress-gan.git
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Train and evaluate your model by following the process in the `impress-gan.ipynb` notebook or by running:
```
python main.py --mode train --num_epochs 150 --model_name my_model
python main.py --mode test --checkpoint_path checkpoint/my_model/checkpoint_epoch_130.pth.tar --model_name my_model
```
4. Deploy your trained model to a web application and make predictions by uploading new images:
```
python app.py
```

## References
- **Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.** A. Radford, L. Metz, S. Chintala. In arXiv, 2015. [Paper](https://arxiv.org/pdf/1511.06434)
- **Improved Techniques for Training GANs.** T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, X. Chen. In arXiv, 2016. [Paper](https://arxiv.org/pdf/1606.03498)

## Acknowledgements
- [Impressionist Landscape Paintings Dataset](https://www.kaggle.com/datasets/robgonsalves/impressionistlandscapespaintings)
- [DCGAN Pytorch Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [CS231N](https://cs231n.stanford.edu/)
- [Pytorch Flask Tutorial](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
