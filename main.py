import torch
import torch.nn as nn
import torch.optim as optim

# ====================================
# 1. Dataset
# ====================================

messages = [
    # Spam
    "win money now",
    "claim your free prize",
    "you won a lottery",
    "free iphone click now",
    "congratulations you won cash",
    "limited offer buy now",
    "earn money quickly",
    "claim your reward now",

    # Not Spam
    "meeting starts at five",
    "please call me later",
    "how are you doing",
    "lets have lunch together",
    "send me the report",
    "see you tomorrow",
    "project submission is today",
    "happy birthday friend"
]

labels = [
    # Spam = 1
    1,1,1,1,1,1,1,1,
    # Not Spam = 0
    0,0,0,0,0,0,0,0
]

# ====================================
# 2. Tokenization
# ====================================

tokenized_messages = [msg.lower().split() for msg in messages]
print("Tokenized Messages:", tokenized_messages)

# ====================================
# 3. Build Vocabulary
# ====================================

vocab = {}
index = 1
for message in tokenized_messages:
    for word in message:
        if word not in vocab:
            vocab[word] = index
            index += 1

print("\nVocabulary:", vocab)

# ====================================
# 4. Word → Index Mapping
# ====================================

sequences = [[vocab[word] for word in message] for message in tokenized_messages]
print("\nSequences:", sequences)

# ====================================
# 5. Padding
# ====================================

max_length = max(len(seq) for seq in sequences)
padded_sequences = [seq + [0]*(max_length-len(seq)) for seq in sequences]
print("\nPadded Sequences:", padded_sequences)

# ====================================
# 6. Convert To Tensors
# ====================================

X = torch.tensor(padded_sequences, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
print("\nInput Shape:", X.shape)

# ====================================
# 7. LSTM Model
# ====================================

class SpamLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        final_hidden = hidden[-1]   # last hidden state
        out = self.fc(final_hidden)
        return self.sigmoid(out)

# ====================================
# 8. Initialize Model
# ====================================

vocab_size = len(vocab) + 1
model = SpamLSTM(vocab_size=vocab_size, embedding_dim=16, hidden_size=32)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ====================================
# 9. Training
# ====================================

epochs = 300
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ====================================
# 10. Prediction Function
# ====================================

def predict_message(text):
    tokens = text.lower().split()
    sequence = [vocab.get(word, 0) for word in tokens]
    sequence += [0]*(max_length - len(sequence))
    tensor = torch.tensor([sequence], dtype=torch.long)

    with torch.no_grad():
        prediction = model(tensor)
        score = prediction.item()

    label = "SPAM" if score >= 0.5 else "NOT SPAM"
    print("\nMessage:", text)
    print("Score:", round(score, 4))
    print("Prediction:", label)

# ====================================
# 11. Testing
# ====================================

predict_message("win free money now")
predict_message("please send the report")
predict_message("claim your reward")
predict_message("see you tomorrow")
