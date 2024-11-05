```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Configuración de dispositivos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformaciones para los datos
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normaliza a [-1, 1]
])

# Cargar los datos de MNIST
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definir el modelo
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 clases para los dígitos 0-9

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Aplanar la imagen
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Inicializar el modelo, la función de pérdida y el optimizador
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar el modelo
def train_model(num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Adelante
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Atrás
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluar el modelo
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')

# Ejecutar el entrenamiento y la evaluación
train_model(num_epochs=5)
evaluate_model()

# Mostrar algunas predicciones
import matplotlib.pyplot as plt

def plot_predictions(images, labels, predictions, num=10):
    plt.figure(figsize=(12, 4))
    for i in range(num):
        plt.subplot(2, 10, i + 1)
        plt.imshow(images[i].cpu().numpy().reshape(28, 28), cmap='gray')
        plt.title(f'True: {labels[i]}, Pred: {predictions[i]}')
        plt.axis('off')
    plt.show()

# Obtener predicciones para mostrar
model.eval()
with torch.no_grad():
    test_images, test_labels = next(iter(test_loader))
    test_images, test_labels = test_images.to(device), test_labels.to(device)
    test_outputs = model(test_images)
    _, test_predictions = torch.max(test_outputs.data, 1)

plot_predictions(test_images, test_labels, test_predictions, num=10)
```

### Explicación del código:
1. **Configuración**: Se establece el dispositivo (CPU o GPU) y se definen las transformaciones para normalizar los datos.
2. **Carga de datos**: Se carga el conjunto de datos MNIST y se crean `DataLoader` para el entrenamiento y la prueba.
3. **Modelo**: Se define una red neuronal simple con capas densas.
4. **Entrenamiento**: Se entrena el modelo en un bucle, calculando la pérdida y actualizando los pesos.
5. **Evaluación**: Se evalúa la precisión del modelo en el conjunto de prueba.
6. **Predicciones**: Se muestran algunas imágenes junto con sus predicciones.