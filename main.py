# VAE intended for unsupervised learning of nifti/mat files of COPD cases
# user discretion is advised

# assume dataset is in MAT files (256x256xDEPTH) or (512x512xDEPTH)
# figure out what 589824 is
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataLoader import train_loader
from dataLoader import test_loader
from train_vae import *
from vqVAE import VQVAE

# Train the VAE for multiple epochs

latent_dim = 32  # Choose the desired dimensionality of the latent space
num_epochs = 10
num_embeddings = 512
embedding_dim = 32
commitment_weight = 0.25

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VQ_MODE = True

    if VQ_MODE == False:
        print(device)
        model = VAE(latent_dim)
        train_vae(model, train_loader, num_epochs, device)
        print("Finished Training, now beginning testing...")
        test_model(model,test_loader,device)
    else:
        print("VQ")
        vq_model = VQVAE(latent_dim=latent_dim, num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_weight=commitment_weight)
        train_vq_vae(vq_model, train_loader, num_epochs, device)
        print("Finished Training, now beginning testing...")
        vq_test_model(vq_model, test_loader, device)