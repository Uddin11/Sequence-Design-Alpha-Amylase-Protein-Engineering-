# Import required packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, backend as K
import matplotlib.pyplot as plt

# Load and preprocess the dataset
data = pd.read_csv(r'C:\Users\shab4\OneDrive - Cardiff University\Desktop\BIg Data Biology\AI_protein_design_exercise\Data\sequences.csv')

# Drop rows with missing values
data_cleaned = data.dropna(subset=['activity_dp7']).copy()

# Define valid amino acids and convert sequences to encoded format
valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
aa_to_int = {aa: i for i, aa in enumerate(valid_amino_acids)}

# Convert amino acids to integer encoding
data_cleaned['encoded_sequence'] = data_cleaned['mutated_sequence'].apply(lambda seq: [aa_to_int[aa] for aa in seq])

# Calculate the sequence length
data_cleaned['sequence_length'] = data_cleaned['mutated_sequence'].apply(len)

# Pad the sequences to ensure uniform length
max_len = data_cleaned['sequence_length'].max()
padded_sequences = pad_sequences(data_cleaned['encoded_sequence'], maxlen=max_len, padding='post')

# Store padded sequences as a new column
data_cleaned['padded_sequence'] = list(padded_sequences)

# Feature engineering for log-transformed activity
data_cleaned['log_activity'] = np.log1p(data_cleaned['activity_dp7'])

# Step 1: Train the Activity Prediction Model
X = np.array(data_cleaned['padded_sequence'].tolist())  
X = np.expand_dims(X, -1)  
y = data_cleaned['log_activity'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the activity prediction model architecture
def build_activity_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build and compile the activity prediction model
activity_model = build_activity_model((X_train.shape[1], X_train.shape[2]))

# Train the model
history = activity_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
test_loss = activity_model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# Save the trained activity model for future use
activity_model.save(r'C:\Users\shab4\OneDrive - Cardiff University\Desktop\BIg Data Biology\AI_protein_design_exercise\new_activity_model.h5')

# Step 2: Variational Autoencoder for Sequence Generation

# VAE Loss function (Reconstruction Loss + KL Divergence)
class VAEModel(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # Compute reconstruction loss
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(inputs, reconstructed))
        reconstruction_loss *= max_len  # Scale based on sequence length

        # Compute KL divergence loss
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        # Add both losses
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        self.add_loss(total_loss)

        return reconstructed

# Encoder Network
def build_encoder(latent_dim):
    inputs = layers.Input(shape=(max_len, len(valid_amino_acids)))
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)

    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    return models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

# Decoder Network
def build_decoder(latent_dim):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation='relu')(latent_inputs)
    x = layers.Dense(max_len * len(valid_amino_acids), activation='relu')(x)
    x = layers.Reshape((max_len, len(valid_amino_acids)))(x)

    outputs = layers.Conv1D(len(valid_amino_acids), 3, padding='same', activation='sigmoid')(x)
    return models.Model(latent_inputs, outputs, name='decoder')

# Build and compile the VAE
latent_dim = 64  # Latent space dimension
encoder = build_encoder(latent_dim)
decoder = build_decoder(latent_dim)
vae = VAEModel(encoder, decoder)
vae.compile(optimizer='adam')

# Train the VAE on the one-hot encoded data
X_train_vae = np.stack(data_cleaned['padded_sequence'].values)
X_train_vae = tf.keras.utils.to_categorical(X_train_vae, num_classes=len(valid_amino_acids))  # One-hot encoding
vae.fit(X_train_vae, epochs=100, batch_size=32)

# Step 3: Generate Diverse Sequences and Predict Activity

# Function to generate random latent vectors for diversity
def generate_random_latent_vectors(num_vectors, latent_dim, variance=1.5):
    """Generates random latent vectors by sampling from a Gaussian distribution."""
    return np.random.normal(size=(num_vectors, latent_dim), scale=variance)

# Function to decode latent vectors to amino acid sequences
def decode_latent_space(latent_vectors, decoder, aa_to_int):
    """Decodes latent vectors into amino acid sequences using the VAE decoder."""
    int_to_aa = {i: aa for aa, i in aa_to_int.items()}
    generated_sequences_aa = []

    for z in latent_vectors:
        z = np.reshape(z, (1, -1))  # Ensure the latent vector has shape (1, 64)
        generated_sequence = decoder.predict(z)[0]  # Decode the latent vector
        decoded_seq = ''.join([int_to_aa[np.argmax(aa)] for aa in generated_sequence])  # Convert one-hot to amino acids
        generated_sequences_aa.append(decoded_seq)

    return generated_sequences_aa

# Generate a large number of random latent vectors
num_sequences = 5000  # Increase this number for more diversity
latent_vectors = generate_random_latent_vectors(num_sequences, latent_dim=64, variance=1.5)

# Decode the generated latent vectors into sequences
generated_sequences = decode_latent_space(latent_vectors, decoder, aa_to_int)

# Predict activity for the generated sequences
generated_encoded_sequences = pad_sequences(
    [[aa_to_int[aa] for aa in seq if aa in aa_to_int] for seq in generated_sequences], 
    maxlen=max_len, padding='post'
)

# Convert to the shape expected by the model
generated_encoded_sequences = np.expand_dims(generated_encoded_sequences, -1)

# Predict activity using the trained activity model
predicted_activity = activity_model.predict(generated_encoded_sequences)

# Step 4: Save Top 100 Best Sequences with Highest Activity

# Combine sequences with their predicted activity
sequence_activity_pairs = list(zip(generated_sequences, predicted_activity.flatten()))

# Sort by predicted activity (in descending order) and take the top 100
top_100_sequences = sorted(sequence_activity_pairs, key=lambda x: x[1], reverse=True)[:100]

# Create a DataFrame to save the top 100 sequences
top_sequences_df = pd.DataFrame(top_100_sequences, columns=['sequence', 'predicted_activity'])

# Save the top 100 sequences to a CSV file
output_file = r'C:\Users\shab4\OneDrive - Cardiff University\Desktop\BIg Data Biology\AI_protein_design_exercise\top_100_sequences.csv'
top_sequences_df.to_csv(output_file, index=False)

print(f"Top 100 sequences saved to {output_file}")


# Generate figures 

# Generate figures

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from tensorflow.keras.utils import plot_model
from sklearn.manifold import TSNE

# Define the output directory
output_dir = r'C:\Users\shab4\OneDrive - Cardiff University\Desktop\BIg Data Biology\AI_protein_design_exercise'

# Ensure the directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Assuming you have 'valid_amino_acids' and 'aa_to_int' defined
valid_amino_acids = sorted(list("ACDEFGHIKLMNPQRSTVWY"))
aa_to_int = {aa: i for i, aa in enumerate(valid_amino_acids)}

# Create a DataFrame for the mapping
encoding_df = pd.DataFrame(list(aa_to_int.items()), columns=['Amino Acid', 'Integer Encoding'])

# Plot the table
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=encoding_df.values, colLabels=encoding_df.columns, loc='center')
table.set_fontsize(14)
table.scale(1, 2)
plt.title('Amino Acid to Integer Encoding', fontsize=16)

# Save the figure
save_path = os.path.join(output_dir, 'amino_acid_encoding_table.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()

# Plotting training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Activity Prediction Model Loss Over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Mean Squared Error Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Save the figure
save_path = os.path.join(output_dir, 'activity_model_loss.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()

# For the encoder
encoder_save_path = os.path.join(output_dir, 'encoder_architecture.png')
plot_model(encoder, to_file=encoder_save_path, show_shapes=True, show_layer_names=True)

# For the decoder
decoder_save_path = os.path.join(output_dir, 'decoder_architecture.png')
plot_model(decoder, to_file=decoder_save_path, show_shapes=True, show_layer_names=True)

# For the entire VAE model
vae_save_path = os.path.join(output_dir, 'vae_architecture.png')
plot_model(vae, to_file=vae_save_path, show_shapes=True, show_layer_names=True)

# Initialize lists to store loss values
vae_loss_history = []

# Custom training loop to record loss
epochs = 100
batch_size = 32
num_batches = int(np.ceil(X_train_vae.shape[0] / batch_size))

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    epoch_loss = 0
    for batch in range(num_batches):
        batch_start = batch * batch_size
        batch_end = min((batch + 1) * batch_size, X_train_vae.shape[0])
        x_batch = X_train_vae[batch_start:batch_end]
        loss = vae.train_on_batch(x_batch, None)
        epoch_loss += loss * x_batch.shape[0]
    epoch_loss /= X_train_vae.shape[0]
    vae_loss_history.append(epoch_loss)
    print(f'Loss: {epoch_loss:.4f}')

# Plotting VAE loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(vae_loss_history, label='VAE Loss')
plt.title('VAE Training Loss Over Epochs', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# Save the figure
save_path = os.path.join(output_dir, 'vae_training_loss.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()

# Encode the input data to get the latent representations
z_mean, z_log_var, z = encoder.predict(X_train_vae)

# Use t-SNE to reduce dimensions to 2D
tsne = TSNE(n_components=2, random_state=42)
z_2d = tsne.fit_transform(z)

# Plot the 2D latent space
plt.figure(figsize=(10, 6))
plt.scatter(z_2d[:, 0], z_2d[:, 1], alpha=0.5)
plt.title('Latent Space Visualization', fontsize=16)
plt.xlabel('Dimension 1', fontsize=14)
plt.ylabel('Dimension 2', fontsize=14)
plt.grid(True)

# Save the figure
save_path = os.path.join(output_dir, 'latent_space_visualization.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()
