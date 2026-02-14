# SMS Spam Classifier - Complete Explanation

## Overview
This solution creates a neural network model to classify SMS messages as either "ham" (legitimate) or "spam" using TensorFlow and Keras.

## Step-by-Step Breakdown

### 1. Data Loading and Preparation
```python
train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_data = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])
```
**What it does:** Loads the TSV (Tab-Separated Values) files into pandas DataFrames with column names 'label' and 'message'.

**Why:** We need structured data to work with. The dataset comes in TSV format where each row has a label (ham/spam) and the message text.

---

### 2. Label Encoding
```python
train_labels = train_data['label'].map({'ham': 0, 'spam': 1}).values
test_labels = test_data['label'].map({'ham': 0, 'spam': 1}).values
```
**What it does:** Converts text labels ('ham', 'spam') to numerical values (0, 1).

**Why:** Neural networks work with numbers, not text. We encode:
- 'ham' → 0 (negative class - not spam)
- 'spam' → 1 (positive class - spam)

---

### 3. Text Tokenization
```python
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_messages)
train_sequences = tokenizer.texts_to_sequences(train_messages)
```
**What it does:**
- Creates a vocabulary of the top 10,000 most common words
- Assigns each word a unique integer ID
- Converts each message into a sequence of integers
- Uses `<OOV>` token for words not in the vocabulary

**Why:** Neural networks can't process raw text. Tokenization converts text into numerical sequences.

**Example:**
```
Original: "win free money now"
Tokenized: [245, 89, 1456, 78]
```

---

### 4. Sequence Padding
```python
train_padded = pad_sequences(train_sequences, maxlen=max_length, 
                             padding='post', truncating='post')
```
**What it does:** Ensures all sequences have the same length (100 tokens) by:
- Adding zeros at the end of shorter sequences
- Cutting off the end of longer sequences

**Why:** Neural networks require fixed-size inputs. Different messages have different lengths.

**Example:**
```
Before: [245, 89, 1456, 78]
After:  [245, 89, 1456, 78, 0, 0, 0, ... 0]  (padded to 100)
```

---

### 5. Model Architecture
```python
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 32, input_length=max_length),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])
```

#### Layer Breakdown:

**a) Embedding Layer (vocab_size=10000, output_dim=32)**
- **Purpose:** Converts word IDs into dense vectors
- **Input:** Integer word IDs (e.g., 245)
- **Output:** 32-dimensional vector representing the word's meaning
- **Why:** Creates semantic relationships between words (similar words have similar vectors)

**b) GlobalAveragePooling1D**
- **Purpose:** Reduces variable-length sequences to fixed-size vectors
- **Input:** Sequence of 32-dimensional vectors (100 x 32)
- **Output:** Single 32-dimensional vector (average of all word vectors)
- **Why:** Summarizes the entire message into one vector

**c) Dense Layer (24 neurons, ReLU activation)**
- **Purpose:** Learns complex patterns in the data
- **Why:** Creates non-linear combinations of features to better distinguish spam from ham

**d) Dropout Layer (0.5)**
- **Purpose:** Randomly deactivates 50% of neurons during training
- **Why:** Prevents overfitting by forcing the network to learn redundant representations

**e) Output Layer (1 neuron, sigmoid activation)**
- **Purpose:** Produces final prediction
- **Output:** Single value between 0 and 1
- **Why:** Sigmoid maps any input to [0,1] range, perfect for binary classification
  - Close to 0 → ham
  - Close to 1 → spam

---

### 6. Model Compilation
```python
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```
**What it does:**
- **Loss function:** Binary cross-entropy (standard for binary classification)
- **Optimizer:** Adam (adaptive learning rate optimizer)
- **Metrics:** Tracks accuracy during training

**Why:** Defines how the model learns and what metric to optimize.

---

### 7. Training
```python
history = model.fit(
    train_padded, 
    train_labels, 
    epochs=10,
    validation_data=(test_padded, test_labels)
)
```
**What it does:** Trains the model for 10 epochs (complete passes through the data).

**Why:** The model learns to distinguish spam from ham by adjusting weights to minimize loss.

---

### 8. Prediction Function
```python
def predict_message(pred_text):
    sequence = tokenizer.texts_to_sequences([pred_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length,
                                    padding='post', truncating='post')
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    label = 'spam' if prediction > 0.5 else 'ham'
    return [float(prediction), label]
```

**What it does:**
1. Tokenizes the input text using the same tokenizer
2. Pads the sequence to length 100
3. Passes it through the model
4. Gets a probability score (0-1)
5. Classifies as 'spam' if probability > 0.5, otherwise 'ham'

**Why:** Applies the same preprocessing pipeline used during training to ensure consistency.

---

## Key Hyperparameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| vocab_size | 10,000 | Number of unique words to keep in vocabulary |
| max_length | 100 | Maximum sequence length |
| embedding_dim | 32 | Dimension of word embeddings |
| hidden_units | 24 | Neurons in hidden layer |
| dropout_rate | 0.5 | Percentage of neurons to drop during training |
| epochs | 10 | Number of complete passes through training data |

---

## Model Performance Tips

### Why This Architecture Works:
1. **Embedding layer** captures semantic meaning of words
2. **Pooling** handles variable-length messages
3. **Dense layer** learns complex spam patterns
4. **Dropout** prevents overfitting on training data
5. **Sigmoid output** provides interpretable probabilities

### Common Spam Indicators the Model Learns:
- Words like "free", "win", "prize", "call now"
- Phone numbers and monetary symbols
- Excessive punctuation (!!!)
- Promotional language patterns

### Typical Performance:
- **Training accuracy:** ~98-99%
- **Validation accuracy:** ~97-98%
- The model should pass all test cases in the challenge

---

## Alternative Approaches (Not Used Here)

1. **LSTM/GRU layers:** Better for sequential patterns but more complex
2. **Bidirectional layers:** Process text in both directions
3. **Pre-trained embeddings:** Use Word2Vec or GloVe instead of learning from scratch
4. **TF-IDF + Traditional ML:** Logistic Regression, SVM (faster but less accurate)

---

## Troubleshooting

**If model doesn't pass tests:**
1. Increase epochs (try 15-20)
2. Adjust vocab_size (try 15,000)
3. Try different max_length (try 120)
4. Add more Dense layers
5. Reduce dropout rate to 0.3

**If model overfits (high training accuracy, low validation):**
1. Increase dropout rate to 0.6
2. Reduce model complexity (fewer neurons)
3. Add regularization (L2)

---

## Code Quality Notes

✅ **What's Good:**
- Clear separation of concerns (data prep, model building, prediction)
- Comprehensive comments explaining each step
- Proper preprocessing pipeline
- Reusable tokenizer and padding configuration
- Robust predict_message function

✅ **Production Improvements:**
- Save and load model weights for reuse
- Add input validation in predict_message
- Implement confidence thresholds
- Add logging for monitoring
- Use callbacks for early stopping

---

## References

- TensorFlow Text Classification Tutorial: https://www.tensorflow.org/tutorials/keras/text_classification
- Keras Sequential API: https://keras.io/guides/sequential_model/
- Understanding Embeddings: https://developers.google.com/machine-learning/crash-course/embeddings
