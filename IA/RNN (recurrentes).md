```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Configuraciones
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Texto de entrenamiento (puedes reemplazarlo con otro texto más grande)
text = "Hola, soy un modelo de lenguaje. Estoy aquí para generar texto. Espero que te guste mi escritura. ¡Vamos a comenzar!"

# Crear un vocabulario
chars = sorted(list(set(text)))
int2char = {i: ch for i, ch in enumerate(chars)}
char2int = {ch: i for i, ch in enumerate(chars)}

# Hiperparámetros
num_epochs = 1000
hidden_size = 128
num_layers = 1
seq_length = 5
learning_rate = 0.01

# Preparar los datos
def prepare_data(text, seq_length):
    inputs = []
    targets = []
    for i in range(0, len(text) - seq_length):
        inputs.append([char2int[ch] for ch in text[i:i + seq_length]])
        targets.append(char2int[text[i + seq_length]])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

X, y = prepare_data(text, seq_length)

# Definir el modelo RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = nn.functional.one_hot(x, num_classes=len(chars)).float()
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Tomamos la última salida
        return out

# Inicializar el modelo, la función de pérdida y el optimizador
model = RNN(input_size=len(chars), hidden_size=hidden_size, output_size=len(chars), num_layers=num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entrenar el modelo
model.train()
for epoch in range(num_epochs):
    inputs, targets = X.to(device), y.to(device)

    optimizer.zero_grad()
    outputs = model(inputs.unsqueeze(1))  # Agregar una dimensión de batch
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generar texto
def generate_text(model, start_str, gen_length):
    model.eval()
    input_str = start_str
    for _ in range(gen_length):
        input_seq = torch.tensor([[char2int[ch] for ch in input_str[-seq_length:]]]).to(device)
        with torch.no_grad():
            output = model(input_seq.unsqueeze(1))
        char_idx = torch.argmax(output).item()
        input_str += int2char[char_idx]
    return input_str

# Generar texto a partir de un texto inicial
start_string = "Hola"
generated_text = generate_text(model, start_string, 50)
print(f"Texto generado: {generated_text}")
```

### Explicación del Código:

1. **Datos de Entrenamiento**: Se define un pequeño texto como fuente de entrenamiento y se crea un vocabulario.
2. **Preparación de Datos**: Se convierte el texto en una secuencia de enteros para usar como entradas y salidas.
3. **Modelo RNN**: Se define una clase para la RNN, que incluye una capa RNN y una capa totalmente conectada (fully connected).
4. **Entrenamiento**: Se entrena el modelo durante un número específico de épocas, minimizando la pérdida de entropía cruzada.
5. **Generación de Texto**: Se genera texto a partir de una cadena inicial, extendiendo la secuencia según lo aprendido por el modelo.