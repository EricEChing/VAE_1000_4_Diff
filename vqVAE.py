import torch.nn as nn
import torch
from vae import VAE
import torch.nn.functional as F


class VQVAE(VAE):
    def __init__(self, latent_dim, num_embeddings, embedding_dim, commitment_weight):
        super(VQVAE, self).__init__(latent_dim)

        # Vector Quantization components
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def vector_quantization(self, z_e):
        distances = torch.norm(
            z_e.unsqueeze(2).unsqueeze(3).unsqueeze(4) - self.embedding.weight.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            dim=1)
        indices = torch.argmin(distances, dim=1)
        quantized_z_e = self.embedding(indices.squeeze())  # Squeeze indices to remove added dimensions
        return quantized_z_e, indices


    def forward(self, x):
        mu, logvar = self.encode(x)
        z_e = self.reparameterize(mu, logvar)

        quantized_z_e, indices = self.vector_quantization(z_e)
        x_hat = self.decode(quantized_z_e)

        # Calculate commitment loss b
        commitment_loss = self.commitment_weight * F.mse_loss(z_e.detach(), quantized_z_e)

        return x_hat, mu, logvar, quantized_z_e, indices, commitment_loss