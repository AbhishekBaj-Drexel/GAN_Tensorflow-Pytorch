import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64

train_data = MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
train_data.data = train_data.data[train_data.targets == 0]
train_data.targets = train_data.targets[train_data.targets == 0]

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(x.size(0), 1, 28, 28)

model_D = Discriminator().to(device)
model_G = Generator().to(device)

criterion = nn.BCELoss()

optimizer_D = optim.Adam(model_D.parameters(), lr=0.0001)
optimizer_G = optim.Adam(model_G.parameters(), lr=0.0001)

EPOCHS = 300
noise_dim = 100

output_dir = './output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

history = {"train_loss_D": [], "train_loss_G": []}

for epoch in range(EPOCHS):
    train_loss_D = 0.0
    train_loss_G = 0.0
    num_batches = 0

    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)
        batch_size = images.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        outputs = model_D(images)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = model_G(z)
        outputs = model_D(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        z = torch.randn(batch_size, noise_dim).to(device)
        fake_images = model_G(z)
        outputs = model_D(fake_images)
        g_loss = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        train_loss_D += d_loss.item()
        train_loss_G += g_loss.item()
        num_batches += 1

    train_loss_D /= num_batches
    train_loss_G /= num_batches

    history["train_loss_D"].append(train_loss_D)
    history["train_loss_G"].append(train_loss_G)

    print(f"Epoch {epoch + 1}/{EPOCHS} - Generator Loss: {train_loss_G:.4f}, Discriminator Loss: {train_loss_D:.4f}")

    with torch.no_grad():
        z = torch.randn(BATCH_SIZE, noise_dim).to(device)
        generated_images = model_G(z).cpu()

        fig = plt.figure(figsize=(10, 10))
        for i in range(BATCH_SIZE):
            plt.subplot(8, 8, i + 1)
            plt.imshow(generated_images[i].view(28, 28), cmap='gray')
            plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'image_at_epoch_{epoch + 1:04d}.png'))
        plt.close(fig)

plt.figure()
plt.plot(history["train_loss_D"], label='Discriminator Loss')
plt.plot(history["train_loss_G"], label='Generator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
plt.show()
