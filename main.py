import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import SGD

latent_dim = 100

# Generator
def build_generator(latent_dim):
    noise = Input(shape=(latent_dim,))
    dense1 = Dense(300, activation='relu')(noise)
    nuclei = Reshape((300,))(dense1)

    dense2 = Dense(3, activation='relu')(noise)
    lattice_params = Reshape((3,))(dense2)

    dense3 = Dense(3, activation='relu')(noise)
    lattice_angles = Reshape((3,))(dense3)

    dense4 = Dense(1, activation='relu')(noise)
    prediction_param = Reshape((1,))(dense4)

    model = Model(inputs=noise, outputs=[nuclei, lattice_params, lattice_angles, prediction_param])
    return model

# Discriminator
def build_discriminator():
    nuclei = Input(shape=(300,))
    lattice_params = Input(shape=(3,))
    lattice_angles = Input(shape=(3,))
    prediction_param = Input(shape=(1,))

    concat = Concatenate(axis=-1)([nuclei, lattice_params, lattice_angles, prediction_param])

    dense1 = Dense(256, activation='relu')(concat)
    dense2 = Dense(128, activation='relu')(dense1)
    validity = Dense(1, activation='sigmoid')(dense2)

    model = Model(inputs=[nuclei, lattice_params, lattice_angles, prediction_param], outputs=validity)
    return model

# Combined model
generator = build_generator(latent_dim)
discriminator = build_discriminator()

optimizer = SGD(0.0002, 0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

noise = Input(shape=(latent_dim,))
nuclei, lattice_params, lattice_angles, prediction_param = generator(noise)

discriminator.trainable = False
validity = discriminator([nuclei, lattice_params, lattice_angles, prediction_param])

combined = Model(inputs=noise, outputs=validity)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

# Fake data
num_samples = 1000
X_nuclei = np.random.rand(num_samples, 300)
X_lattice_params = np.random.rand(num_samples, 3)
X_lattice_angles = np.random.rand(num_samples, 3)
X_prediction_param = np.random.rand(num_samples, 1)

data = (X_nuclei, X_lattice_params, X_lattice_angles, X_prediction_param)

# Training function
def train(generator, discriminator, combined, data, epochs, batch_size=64):
    X_nuclei, X_lattice_params, X_lattice_angles, X_prediction_param = data
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train the discriminator
        idx = np.random.randint(0, X_nuclei.shape[0], batch_size)
        real_nuclei = X_nuclei[idx]
        real_lattice_params = X_lattice_params[idx]
        real_lattice_angles = X_lattice_angles[idx]
        real_prediction_param = X_prediction_param[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_nuclei, gen_lattice_params, gen_lattice_angles, gen_prediction_param = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch([real_nuclei, real_lattice_params, real_lattice_angles, real_prediction_param], valid)
        d_loss_fake = discriminator.train_on_batch([gen_nuclei, gen_lattice_params, gen_lattice_angles, gen_prediction_param], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = combined.train_on_batch(noise, valid)

        #if epoch % 100 == 0:
        print("Epoch: %d, D loss: %f, G loss: %f" % (epoch, d_loss[0], g_loss))

# Train the GAN
train(generator, discriminator, combined, data, epochs=100, batch_size=64)
