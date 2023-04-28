import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, TimeDistributed, Masking, Concatenate, Lambda
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras.optimizers import Adam
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

latent_dim = 100
max_nuclei = 50  # Maximum number of nuclei

def build_discriminator():

    nuclei_input = Input(shape=(None, 4))
    masked_nuclei = Masking(mask_value=0.0)(nuclei_input)
    x_nuclei = Dense(32, activation='relu')(masked_nuclei)
    x_nuclei = Lambda(lambda x: tf.reduce_sum(x, axis=1))(x_nuclei)

    lattice_params_input = Input(shape=(3,))
    lattice_angles_input = Input(shape=(3,))
    prediction_param_input = Input(shape=(1,))

    x = Concatenate()([x_nuclei, lattice_params_input, lattice_angles_input, prediction_param_input])
    x = Dense(512, activation='relu')(x)
    validity = Dense(1, activation='sigmoid')(x)

    return Model(inputs=[nuclei_input, lattice_params_input, lattice_angles_input, prediction_param_input], outputs=validity)


def build_generator():
    noise = Input(shape=(latent_dim,))
    x = Dense(512, activation='relu')(noise)
    x = Dense(max_nuclei * 4, activation='relu')(x)
    nuclei_output = Reshape((max_nuclei, 4))(x)

    lattice_params_output = Dense(3, activation='linear')(x)
    lattice_angles_output = Dense(3, activation='linear')(x)
    prediction_param_output = Dense(1, activation='linear')(x)

    return Model(inputs=noise, outputs=[nuclei_output, lattice_params_output, lattice_angles_output, prediction_param_output])

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

generator = build_generator()
z = Input(shape=(latent_dim,))
nuclei, lattice_params, lattice_angles, prediction_param = generator(z)
discriminator.trainable = False
validity = discriminator([nuclei, lattice_params, lattice_angles, prediction_param])
combined = Model(z, validity)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training loop
batch_size = 64
epochs = 200

X_nuclei = np.random.random((1000, max_nuclei, 4))
X_lattice_params = np.random.random((1000, 3))
X_lattice_angles = np.random.random((1000, 3))
X_prediction_param = np.random.random((1000, 1))
y = np.ones((batch_size, 1))

for epoch in range(epochs):
    idx = np.random.randint(0, X_nuclei.shape[0], batch_size)
    real_nuclei = X_nuclei[idx]
    real_lattice_params = X_lattice_params[idx]
    real_lattice_angles = X_lattice_angles[idx]
    real_prediction_param = X_prediction_param[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_nuclei, gen_lattice_params, gen_lattice_angles, gen_prediction_param = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch([real_nuclei, real_lattice_params, real_lattice_angles, real_prediction_param], y)
    d_loss_fake = discriminator.train_on_batch([gen_nuclei, gen_lattice_params, gen_lattice_angles, gen_prediction_param], np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = combined.train_on_batch(noise, y)

    #if epoch % 1000 == 0:
    print(f"Epoch {epoch}: D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]} | G loss: {g_loss}")
