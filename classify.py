import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformer import TokenAndPositionEmbedding, TransformerBlock

vocab_size = 200000
maxlen = 200
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words =vocab_size)

print(len(x_train), " Training Sequences.")
print(len(x_val), " Validation Sequences.")
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

embed_dim = 32
num_heads = 2
ff_dim = 32

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
X = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
X = transformer_block(X)
X = layers.GlobalAvgPool1D()(X)
X = layers.Dropout(0.1)(X)
X = layers.Dense(20, activation='relu')(X)
X = layers.Dropout(0.1)(X)
outputs = layers.Dense(2, activation='softmax')(X)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))
print("Done")






