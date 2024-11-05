```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parámetros de la red
latent_size = 64
hidden_size = 256
num_epochs = 100
batch_size = 100
learning_rate = 0.0002

# Cargar el conjunto de datos MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

# Definir el modelo del Generador
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 28 * 28),
            nn.Tanh()  # Salida entre -1 y 1
        )

    def forward(self, x):
        return self.model(x).view(-1, 1, 28, 28)

# Definir el modelo del Discriminador
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Salida entre 0 y 1
        )

    def forward(self, x):
        return self.model(x.view(-1, 28 * 28))

# Inicializar el generador y el discriminador
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Definir la función de pérdida y los optimizadores
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Entrenar el GAN
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # Cargar imágenes reales y preparar etiquetas
        images = images.to(device)
        real_labels = torch.ones(images.size(0), 1).to(device)
        fake_labels = torch.zeros(images.size(0), 1).to(device)

        # Entrenar el discriminador
        optimizer_d.zero_grad()
        outputs = discriminator(images)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(images.size(0), latent_size).to(device)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Entrenar el generador
        optimizer_g.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)  # Queremos que el generador produzca imágenes reales
        g_loss.backward()
        optimizer_g.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

# Generar imágenes después del entrenamiento
def generate_images(generator, num_images):
    z = torch.randn(num_images, latent_size).to(device)
    fake_images = generator(z)
    return fake_images

# Visualizar algunas imágenes generadas
num_images = 16
generated_images = generate_images(generator, num_images)

# Configurar la visualización
plt.figure(figsize=(8, 8))
for i in range(num_images):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i].detach().cpu().squeeze(), cmap='gray')
    plt.axis('off')
plt.show()
```

### Explicación del Código:

1. **Carga de Datos**: Se utiliza el conjunto de datos MNIST y se aplican transformaciones para normalizar las imágenes.
2. **Modelo Generador**: Se define una red que toma un vector de ruido y lo transforma en una imagen.
3. **Modelo Discriminador**: Se define una red que clasifica las imágenes como reales o falsas.
4. **Entrenamiento**: Se entrena el discriminador y el generador alternadamente:
   - El discriminador se entrena con imágenes reales y generadas.
   - El generador se entrena para engañar al discriminador haciendo que las imágenes generadas se clasifiquen como reales.
5. **Generación de Imágenes**: Después del entrenamiento, se generan nuevas imágenes utilizando el generador.
6. **Visualización**: Se muestran algunas imágenes generadas en una cuadrícula.

### Notas:
- Este es un ejemplo básico y se puede mejorar de muchas maneras, como ajustar los hiperparámetros, agregar técnicas de estabilización de entrenamiento, o utilizar un conjunto de datos más complejo como CelebA para generar rostros.