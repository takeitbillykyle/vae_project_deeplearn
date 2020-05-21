from keras.layers import Lambda, Input, Dense, Activation 
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

sess = tf.compat.v1.InteractiveSession()


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]
    print(batch,dim)
    epsilon = K.random_normal(shape = (batch,dim))
    z_sampled = z_mean + K.exp(0.5*z_log_var)*epsilon
    
    return z_sampled

#Global parameters
input_shape = (14*32,)
encoder_layers_dim = [64,32]
encoder_layers_activations = ['relu','relu']
num_encoder_layers = len(encoder_layers_dim)
decoder_layers_dim = [32,64]
decoder_layers_activations = ['relu','relu']
num_decoder_layers = len(decoder_layers_dim)
latent_dim = 4
# Build VAE-model the keras-way

#ENCODER
inputs = Input(shape = input_shape, name = 'encoder_input')
previous_layer = inputs
for i in range(num_encoder_layers):
    x = Dense(encoder_layers_dim[i], activation = encoder_layers_activations[i])(previous_layer)
    previous_layer = x
    
#LATENT
z_mean = Dense(latent_dim, name='z_mean')(previous_layer)
z_log_var = Dense(latent_dim, name='z_log_var')(previous_layer)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

#DECODER
latent_inputs = Input(shape = (latent_dim,), name = 'z_sampling')
previous_layer = latent_inputs
for i in range(num_decoder_layers):
    x = Dense(decoder_layers_dim[i], activation = decoder_layers_activations[i])(previous_layer)
    previous_layer = x    
outputs = Dense(input_shape[0], activation = 'sigmoid')(previous_layer)
# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs_vae = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs_vae, name='vae_mlp')

'''
beta = 1
reconstruction_loss = binary_crossentropy(inputs, outputs_vae)     
reconstruction_loss *= input_shape[0]     #OBS WHY??
kl_loss = 1 + z_log_var - z_mean**2 - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5*beta
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
x_train0 = np.load('data/data_matrix0.npy')
x_train1 = np.load('data/data_matrix1.npy')
x_train2 = np.load('data/data_matrix2.npy')
x_train3 = np.load('data/data_matrix3.npy')
x_train4 = np.load('data/data_matrix4.npy')
x_train5 = np.load('data/data_matrix5.npy')
x_train = np.concatenate((x_train0,x_train1,x_train2,x_train3,x_train4,x_train5))
x_test = np.load('data/data_matrix6.npy')
epochs = 10
batch_size = 64
#vae.fit(x_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test, None))
#vae.save_weights('vae_100e_64b.h5')
'''
vae.load_weights('vae_100e_32b.h5')
KMP_DUPLICATE_LIB_OK=True
samples = np.random.normal(0,1, size = (10,4))
t = decoder.predict(samples)
t = t.reshape(10,14,32)
for i in range(10):
    print()
    print("hidden values")
    print(samples[i])
    print("activation max")
    print(np.max(t[i].flatten()))
    print("activation min")
    print(np.min(t[i].flatten()))
    print()
    plt.matshow(t[i])
    plt.show()




