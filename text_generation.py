from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
import time

# Required for eager execution to use take operation.
tf.enable_eager_execution()

# We will read every single file as text in this path. It's relative to execution
# Give only folder name
MAIN_PARAMS = { "local": True,
                "source_path": "ataturk",
                "remote": ["shakespeare.txt","https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"],
                "sequence_length": 100,
                "batch_size": 64,
                "buffer_size": 10000,
                "embedding_dimensions": 256,
                "rnn_unit_count": 1024,
                "epoch_count": 50,
                "generation_length": 2000
}

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def generate_text(model, start_string, char2idx, idx2char):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = MAIN_PARAMS['generation_length']

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

def execution():

    if MAIN_PARAMS['local']:
        text = ""
        for filename in os.listdir(MAIN_PARAMS['source_path']):
            text += open(os.path.join(MAIN_PARAMS['source_path'], filename), encoding="utf-8").read()
    else:
        path_to_file = tf.keras.utils.get_file(MAIN_PARAMS['remote'][0], MAIN_PARAMS['remote'][1])
        # Read, then decode for py2 compat.
        text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    print ('Length of text: {} characters'.format(len(text)))
    # The unique characters in the file
    vocab = sorted(set(text))
    print ('{} unique characters'.format(len(vocab)))

    # Creating a mapping from unique characters to indices
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])

    # The maximum length sentence we want for a single input in characters
    seq_length = MAIN_PARAMS['sequence_length']
    examples_per_epoch = len(text)//(seq_length+1)

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    dataset = sequences.map(split_input_target)
    BATCH_SIZE = MAIN_PARAMS['batch_size']
    BUFFER_SIZE = MAIN_PARAMS['buffer_size']

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Length of the vocabulary in chars
    vocab_size = len(vocab)
    embedding_dim = MAIN_PARAMS['embedding_dimensions']
    rnn_units = MAIN_PARAMS['rnn_unit_count']

    model = build_model(
      vocab_size = len(vocab),
      embedding_dim=embedding_dim,
      rnn_units=rnn_units,
      batch_size=BATCH_SIZE)

    print("Model summary:")
    model.summary()
    print("Compiling model. Optimizer: Adam")
    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS=MAIN_PARAMS['epoch_count']

    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

    tf.train.latest_checkpoint(checkpoint_dir)

    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    model.build(tf.TensorShape([1, None]))

    generated_text = generate_text(model, start_string=u"Efendiler", char2idx=char2idx, idx2char=idx2char)

    f = open("generated_output.txt", "a",  encoding="utf-8")
    f.write(generated_text)
    f.close()

    print(generated_text)

# Start execution
execution();
