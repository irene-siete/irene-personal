```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generar datos sintéticos
np.random.seed(42)
time_steps = np.linspace(0, 100, num=500)
data = np.sin(time_steps) + np.random.normal(0, 0.1, size=time_steps.shape)

# Preparar los datos para LSTM
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

seq_length = 10
X, y = create_sequences(data, seq_length)

# Convertir a tensores
X = torch.FloatTensor(X).view(-1, seq_length, 1).to(device)  # Formato [batch, seq_length, features]
y = torch.FloatTensor(y).view(-1, 1).to(device)  # Formato [batch, features]

# Definir el modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Usar la última salida
        return out

# Inicializar el modelo, la función de pérdida y el optimizador
input_size = 1
hidden_size = 64
num_layers = 1
output_size = 1

model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entrenar el modelo
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Hacer predicciones
model.eval()
with torch.no_grad():
    predictions = model(X).cpu().numpy()

# Visualizar los resultados
plt.figure(figsize=(12, 6))
plt.plot(time_steps, data, label='Datos Originales')
plt.plot(time_steps[seq_length:], predictions, label='Predicciones LSTM', color='red')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.show()
```

### Explicación del Código:

1. **Generación de Datos Sintéticos**: Se crea una serie temporal usando una función seno con ruido para simular datos reales.
2. **Preparación de Datos**: Se crean secuencias de datos para que el modelo LSTM pueda aprender. Cada secuencia tiene una longitud de `seq_length`, y el modelo predice el siguiente valor en la serie.
3. **Definición del Modelo LSTM**: Se define una clase `LSTMModel` que incluye una capa LSTM y una capa totalmente conectada (fully connected).
4. **Entrenamiento**: Se entrena el modelo durante un número determinado de épocas, minimizando la pérdida cuadrática media (MSE).
5. **Predicciones**: Se realizan predicciones sobre el conjunto de datos.
6. **Visualización**: Se visualizan los datos originales y las predicciones del modelo.

### Notas:
- Puedes ajustar el número de épocas, el tamaño del lote y otros hiperparámetros para mejorar el rendimiento.
- Para datos reales, asegúrate de preprocesar adecuadamente tus datos, normalizarlos y dividirlos en conjuntos de entrenamiento y prueba.