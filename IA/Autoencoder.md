```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parámetros
batch_size = 128
learning_rate = 0.001
num_epochs = 10

# Transformación de los datos
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Cargar el conjunto de datos MNIST
mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

# Definir el modelo del Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Codificación a 32 dimensiones
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Tanh()  # Salida entre -1 y 1
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Aplanar la imagen
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)

# Inicializar el modelo, la función de pérdida y el optimizador
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entrenar el modelo
for epoch in range(num_epochs):
    for images, _ in data_loader:
        images = images.to(device)

        # Agregar ruido a las imágenes
        noisy_images = images + 0.2 * torch.randn_like(images)
        noisy_images = torch.clamp(noisy_images, 0., 1.)  # Limitar a [0, 1]

        # Entrenar el autoencoder
        optimizer.zero_grad()
        outputs = model(noisy_images)
        loss = criterion(outputs, images)  # Compara con la imagen original
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Visualizar algunas imágenes originales, ruidosas y reconstruidas
def visualize_results(original, noisy, reconstructed):
    plt.figure(figsize=(12, 6))

    # Mostrar imágenes originales
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(original.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')

    # Mostrar imágenes ruidosas
    plt.subplot(1, 3, 2)
    plt.title("Noisy")
    plt.imshow(noisy.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')

    # Mostrar imágenes reconstruidas
    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    plt.imshow(reconstructed.squeeze().cpu().numpy(), cmap='gray')
    plt.axis('off')

    plt.show()

# Probar el autoencoder con una imagen de prueba
test_images, _ = next(iter(data_loader))
test_images = test_images.to(device)
noisy_test_images = test_images + 0.2 * torch.randn_like(test_images)
noisy_test_images = torch.clamp(noisy_test_images, 0., 1.)

with torch.no_grad():
    reconstructed_images = model(noisy_test_images)

# Visualizar los resultados para la primera imagen del lote
visualize_results(test_images[0], noisy_test_images[0], reconstructed_images[0])
```

### Explicación del Código:

1. **Carga de Datos**: Se utiliza el conjunto de datos MNIST y se aplican transformaciones para normalizar las imágenes.
2. **Definición del Modelo Autoencoder**: Se define una clase `Autoencoder` que incluye una red para codificar (encoder) y decodificar (decoder) las imágenes.
3. **Entrenamiento**: Durante el entrenamiento, se añade ruido a las imágenes para simular el proceso de eliminación de ruido. El modelo intenta reconstruir la imagen original a partir de la versión ruidosa.
4. **Visualización**: Después del entrenamiento, se visualizan las imágenes originales, ruidosas y las reconstruidas para evaluar el rendimiento del autoencoder.

### Notas:
- Puedes ajustar la arquitectura del modelo, el tamaño del lote y el número de épocas para obtener mejores resultados.
- Para datos reales, asegúrate de preprocesar adecuadamente tus datos antes de entrenar el modelo.