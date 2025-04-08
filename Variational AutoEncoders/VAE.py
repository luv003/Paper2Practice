import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

class VAEConfig:
    def __init__(self, batch_size=32, epochs=100, latent_dim=200, hidden_dim=400, lr=1e-4, x_dim=784, device=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.x_dim = x_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky_relu(self.fc_input(x))
        x = self.leaky_relu(self.fc_hidden(x))
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, z):
        z = self.leaky_relu(self.fc_hidden(z))
        x_hat = torch.sigmoid(self.fc_output(z))
        return x_hat


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mean, log_var):
        epsilon = torch.randn_like(log_var).to(mean.device)
        z = mean + torch.exp(0.5 * log_var) * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var


class VAETrainer:
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config.lr)
        self.bce_loss = nn.BCELoss(reduction='sum')

    def loss_function(self, x, x_hat, mean, log_var):
        # Binary Cross-Entropy Loss
        bce_loss = self.bce_loss(x_hat, x.view(-1, self.config.x_dim))
        # Kullback-Leibler Divergence Loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return bce_loss + kl_loss

    def train(self):
        self.model.train()
        for epoch in range(self.config.epochs):
            total_loss = 0
            for batch_idx, (x, _) in enumerate(self.train_loader):
                x = x.view(self.config.batch_size, -1).to(self.config.device)

                self.optimizer.zero_grad()
                x_hat, mean, log_var = self.model(x)
                loss = self.loss_function(x, x_hat, mean, log_var)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.config.epochs} - Average Loss: {total_loss / len(self.train_loader.dataset)}")

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(tqdm(self.test_loader)):
                x = x.view(self.config.batch_size, -1).to(self.config.device)
                x_hat, _, _ = self.model(x)
                self.display_images(x, x_hat)
                break

    def display_images(self, original, reconstructed, idx=0):
        def show_image(x, idx):
            x = x.view(self.config.batch_size, 28, 28)
            plt.figure()
            plt.imshow(x[idx].cpu().numpy(), cmap='gray')
            plt.show()

        show_image(original, idx)
        show_image(reconstructed, idx)
        
    def generate_images(self):
        self.model.eval()
        with torch.no_grad():
            noise = torch.randn(self.config.batch_size, self.config.latent_dim).to(self.config.device)
            generated_images = self.model.decoder(noise)
            save_image(generated_images.view(self.config.batch_size, 1, 28, 28), 'generated_images.png')
            self.display_images(generated_images, generated_images, idx=0)
            

def load_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_dataset = MNIST(root='~/datasets', transform=transform, train=True, download=True)
    test_dataset = MNIST(root='~/datasets', transform=transform, train=False, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def main():
    config = VAEConfig(batch_size=32, epochs=100, latent_dim=200, hidden_dim=400, lr=1e-4, x_dim=784)
    train_loader, test_loader = load_data(config.batch_size)

    encoder = Encoder(input_dim=config.x_dim, hidden_dim=config.hidden_dim, latent_dim=config.latent_dim)
    decoder = Decoder(latent_dim=config.latent_dim, hidden_dim=config.hidden_dim, output_dim=config.x_dim)

    vae_model = VAE(encoder, decoder).to(config.device)

    trainer = VAETrainer(vae_model, train_loader, test_loader, config)

    print("Training VAE...")
    trainer.train()

    print("Evaluating VAE...")
    trainer.evaluate()

    print("Generating New Samples...")
    trainer.generate_images()


if __name__ == '__main__':
    main()
