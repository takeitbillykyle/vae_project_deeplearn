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

arc = 4

if arc == 4:
    arc_name = "4_latent_512_256_100e_64b.h5"
    root = "data/latent_and_output_h4/"
if arc == 20:
    arc_name = "20_latent_256_128_100e_64b.h5"
    root = "data/latent_and_output_h20/"

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

if arc == 4:
    input_shape = (14*32,)
    encoder_layers_dim = [512,256]
    encoder_layers_activations = ['relu','relu']
    num_encoder_layers = len(encoder_layers_dim)
    decoder_layers_dim = [256,512]
    decoder_layers_activations = ['relu','relu']
    num_decoder_layers = len(decoder_layers_dim)
    latent_dim = 4

elif arc == 20:
    input_shape = (14*32,)
    encoder_layers_dim = [256,128]
    encoder_layers_activations = ['relu','relu']
    num_encoder_layers = len(encoder_layers_dim)
    decoder_layers_dim = [128,256]
    decoder_layers_activations = ['relu','relu']
    num_decoder_layers = len(decoder_layers_dim)
    latent_dim = 20

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


beta = 1
reconstruction_loss = binary_crossentropy(inputs, outputs_vae)     
reconstruction_loss *= input_shape[0]     #OBS WHY??
kl_loss = 1 + z_log_var - z_mean**2 - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5*beta
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
x_train_sets = []

for i in range(7):
    x_train_sets.append( np.load('data/data_matrix' + str(i) + ".npy"))

x_train0 = x_train_sets[0]
x_train1 = x_train_sets[1]
x_train2 = x_train_sets[2]
x_train3 = x_train_sets[3]
x_train4 = x_train_sets[4]
x_train5 = x_train_sets[5]

x_train = np.concatenate((x_train0,x_train1,x_train2,x_train3,x_train4,x_train5))
x_test = np.load('data/data_matrix6.npy')
epochs = 10
batch_size = 64

#vae.fit(x_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test, None))
#vae.save_weights('vae_100e_64b.h5')

vae.load_weights(arc_name)

latent_name_root = root + "latent_matrix_"

if False:
    for i in range(len(x_train_sets)):
        prediction = encoder.predict([x_train_sets[i]])
        print(prediction)
        latent = prediction[2]
        print("length of predict return")
        print(len(prediction))
        print("shape of latent " + str(i))
        print(latent.shape)
        np.save(latent_name_root + str(i) + "_h" + str(arc) + ".npy", latent)

limit = 5
correct = 0

output_activation_list = []

if False:
    for i in range(len(x_train_sets)):
        latent_activation = encoder.predict([x_train_sets[i]])[2]
        output_activation = decoder.predict(latent_activation)
        output_activation_list.append(output_activation)
        remainder = x_train_sets[i] - np.round(output_activation)

        for j in range(remainder.shape[0]):
            error = np.sum(np.abs(remainder[j,:]))

            if error < limit:
                correct += 1

    np.save(root + "output_activations_h" + str(arc) + ".npy", np.array(output_activation_list))

    print("correct guesses")
    print(correct)
    print("hit rate")
    print(correct/(5000*6))


if True:
    hip_hop = np.load("data/labeled_data/hip_hop.npy")
    metal = np.load("data/labeled_data/metal.npy")
    rock = np.load("data/labeled_data/rock.npy")
    funk = np.load("data/labeled_data/funk.npy")

    collection = [hip_hop,metal,rock,funk]

    latents = []
    for inputs in collection:
        for i in range(int(inputs.shape[0]/14)):
            pattern = inputs[i*14:(i + 1) * 14].reshape(1,448)
            latent = encoder.predict([pattern])[2]
            latents.append(latent)
    
    latents = np.array(latents)
    print("shape of latents")
    print(latents.shape)
    np.save("data/labeled_data/reps_h" + str(arc) + "/representations.npy", latents)





