```python
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datos de ejemplo (oraciones en un "idioma" simple)
input_sentences = ["hola", "mundo", "me gusta", "piano"]
target_sentences = ["hello", "world", "I like", "piano"]

# Crear un vocabulario
input_vocab = {word: idx for idx, word in enumerate(set(" ".join(input_sentences).split()))}
target_vocab = {word: idx for idx, word in enumerate(set(" ".join(target_sentences).split()))}
input_vocab_size = len(input_vocab)
target_vocab_size = len(target_vocab)

# Hiperparámetros
embedding_dim = 10
hidden_size = 16
num_epochs = 100
learning_rate = 0.01

# Modelo de Atención
class Attention(nn.Module):
    def forward(self, decoder_hidden, encoder_outputs):
        attention_weights = torch.bmm(decoder_hidden.unsqueeze(1), encoder_outputs.permute(0, 2, 1))
        attention_weights = torch.softmax(attention_weights, dim=-1)
        context_vector = torch.bmm(attention_weights, encoder_outputs)
        return context_vector, attention_weights

# Modelo Seq2Seq con Atención
class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.decoder_embedding = nn.Embedding(target_vocab_size, embedding_dim)
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.attention = Attention()
        self.fc = nn.Linear(hidden_size, target_vocab_size)

    def forward(self, input_tensor, target_tensor):
        encoder_outputs, (hidden, cell) = self.encoder_lstm(self.encoder_embedding(input_tensor))
        
        # Inicializar el estado del decodificador
        outputs = []
        for t in range(target_tensor.size(1)):
            context_vector, _ = self.attention(hidden[-1], encoder_outputs)
            decoder_input = self.decoder_embedding(target_tensor[:, t]).unsqueeze(1)
            decoder_input = torch.cat((decoder_input, context_vector), dim=2)
            output, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            output = self.fc(output)
            outputs.append(output)

        return torch.cat(outputs, dim=1)

# Preparar los datos
def prepare_data(sentences, vocab):
    data = []
    for sentence in sentences:
        indices = [vocab[word] for word in sentence.split()]
        data.append(torch.tensor(indices).unsqueeze(0))  # Añadir dimensión de batch
    return torch.cat(data)

input_data = prepare_data(input_sentences, input_vocab).to(device)
target_data = prepare_data(target_sentences, target_vocab).to(device)

# Inicializar el modelo, la función de pérdida y el optimizador
model = Seq2Seq(input_vocab_size, target_vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entrenamiento
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    output = model(input_data, target_data[:, :-1])
    loss = criterion(output.view(-1, target_vocab_size), target_data[:, 1:].view(-1))
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Ejemplo de predicción
def translate(sentence):
    model.eval()
    with torch.no_grad():
        input_tensor = prepare_data([sentence], input_vocab).to(device)
        output = model(input_tensor, torch.zeros((1, 4), dtype=torch.long).to(device))  # Tamaño de la secuencia objetivo
        _, predicted_indices = torch.max(output, dim=2)
        translated_sentence = " ".join([list(target_vocab.keys())[idx] for idx in predicted_indices[0].cpu().numpy()])
    return translated_sentence

# Traducir una oración de ejemplo
print("Traducción:", translate("hola mundo"))
```

### Explicación del Código:

1. **Datos de Ejemplo**: Se definen algunas oraciones simples en un "idioma" de ejemplo, junto con sus traducciones.
2. **Vocabulario**: Se crea un vocabulario para las oraciones de entrada y salida, asignando índices a cada palabra.
3. **Modelo de Atención**: Se define una clase `Attention` que calcula el vector de contexto basado en las salidas del codificador y el estado oculto del decodificador.
4. **Modelo Seq2Seq**: Se define el modelo que incluye un codificador y un decodificador. Utiliza la atención para mejorar el rendimiento de la traducción.
5. **Preparación de Datos**: Se preparan los datos de entrada y salida como tensores de PyTorch.
6. **Entrenamiento**: Se entrena el modelo utilizando la función de pérdida de entropía cruzada.
7. **Predicción**: Se define una función para traducir oraciones utilizando el modelo entrenado.

### Notas:
- Este ejemplo es simplificado y puede no ser representativo de un modelo de traducción de lenguajes real. En aplicaciones más complejas, se usan conjuntos de datos mucho más grandes y técnicas avanzadas.
- Puedes modificar el tamaño del vocabulario, la longitud de las oraciones y los hiperparámetros para ver cómo afectan al rendimiento del modelo.