import math, time, csv, torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(111)
train_data_length = 30000

i = 0
train_foils = []
with open("snippy.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        i += 1
        rowFloats = list(map(float, row))
        train_foils.append(rowFloats)
        if i == train_data_length:
            break

n = np.array(train_foils)
train_data = torch.as_tensor(train_foils)
train_labels = torch.zeros(train_data_length)
train_set = [ (train_data[j], train_labels[j]) for j in range(train_data_length)]

#Preparing data and model
batch_size = 30
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
print(train_loader)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

    def forward(self, x):
        output = self.model(x)
        return output

discriminator = Discriminator()
generator = Generator()

lr = 0.001
num_epochs = 500
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

#Training Loop
for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples_labels = torch.ones((batch_size, 1))
        latent_space_samples = torch.randn(batch_size, 8)
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn(batch_size, 8)

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        if epoch % 10 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

latent_space_sample = torch.as_tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.9256, 0.15299, -0.155]])
generated_sample = generator(latent_space_sample)
generated_sample = generated_sample.detach()
print(generated_sample)