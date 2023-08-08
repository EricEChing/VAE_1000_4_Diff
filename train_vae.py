import torch.optim as optim
import matplotlib.pyplot as plt

from dataLoader import valid_loader
from vae_loss import *
from vae_loss import validation_Model


# Training loop
def train_vae(model, train_loader, num_epochs, device):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, num_epochs + 1):
        train_loss = 0
        num_batches = 0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)  # If needed, data is in tensor form here
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(y_pred=recon_batch, y_true=data, mu=mu, log_var=logvar)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            num_batches = num_batches + 1
            print('Completed ' + str(num_batches) + ' batch...')
        print('Epoch: {} Average Loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

        #Find the validation loss:
        current_loss = validation_Model(
            model, valid_loader, vae_loss, device)
        print('Epoch: {} Average Validation Loss: {:.4f}'.format(epoch, current_loss))

def train_vq_vae(model, train_loader, num_epochs, device):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, num_epochs + 1):
        train_loss = 0
        num_batches = 0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)  # If needed, data is in tensor form here
            optimizer.zero_grad()
            x_hat, mu, logvar, quantized_z_e, indices, commitment_loss = model(data)
            loss = vq_vae_loss(y_pred=x_hat, y_true=data, mu=mu, log_var=logvar,commitment_loss=commitment_loss)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            num_batches = num_batches + 1
            print('Completed ' + str(num_batches) + ' batch...')
        print('Epoch: {} Average Loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

        #Find the validation loss:
        current_loss = vq_validation_Model(
            model, valid_loader, vae_loss, device)
        print('Epoch: {} Average Validation Loss: {:.4f}'.format(epoch, current_loss))
