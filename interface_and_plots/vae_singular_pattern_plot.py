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
from matplotlib.pyplot import figure

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

plt.rc('font', **font)

arc = 20

if arc == 4:
    arc_name = "4_latent_512_256_100e_64b.h5"
    root = "data/latent_and_output_h4/"
if arc == 20:
    arc_name = "20_latent_256_128_100e_64b.h5"
    root = "data/latent_and_output_h20/"

hid1 = np.array([[-0.31388889, -0.85833333, -0.95555556, -0.96944444, -0.68055556, -0.92777778,
   0.86944444,  0.95277778, -0.81944444,  0.725  ,     0.76944444, -0.975,
  -0.775,       0.11944444, -0.46944444, -0.29166667, -0.63055556,  0.25277778,
   0.75277778, -0.375     ]])

hid2 = np.array([[[-0.31388889, -0.85833333,  0.9,        -0.92222222, -0.75277778, -0.575,
   0.71944444,  0.38055556, -0.28611111, -0.96944444,  0.93888889, -0.925,
   0.81111111, -0.71944444, -0.51111111,  0.38055556, -0.63055556,  0.25277778,
   0.75277778, -0.375,     ]]])

hid3 = np.array([[ 0.37777778, -0.52777778, -0.56944444, -0.64722222, -0.375,      -0.53888889,
  -0.40833333, -0.6 ,       -0.40555556, -0.10277778, -0.25,        0.48333333,
   0.41944444, -0.37777778,  0.88888889,  0.45833333, -0.025,       0.38888889,
  -0.33333333, -0.25      ]])

hid4 = np.array([[ 0.24444444, -0.53888889,  0.13055556, -0.71944444,  0.58888889, -0.47222222,
   0.91111111,  0.31388889, -0.55  ,     -0.58888889, -0.56944444,  0.37222222,
   0.41944444, -0.37777778,  0.53888889,  0.35833333, -0.02777778,  0.71666667,
   0.33611111, -0.43888889]])

hids = [hid1,hid2,hid3,hid4]

if True:
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

    hiddens = np.array([[0,0,-0.5,-1],[0,0,-0.5,1],[0,0,0.5,-1],[0,0,0.5,1]])
    outs = np.zeros((4,448))

    for i in range(hiddens.shape[0]):
        outs[i] = decoder.predict(hiddens[i].reshape(1,4))

    np.save("data/latent_and_output_h4/entangle_example_outs.npy",outs)

if False:

    hiddens = ("h1 = 0, h2 = 0, h3 = -0.5, h4 = -1","h1 = 0, h2 = 0, h3 = -0.5, h4 = 1","h1 = 0, h2 = 0, h3 = 0.5, h4 = -1","h1 = 0, h2 = 0, h3 = 0.5, h4 = 1",)
    outs = np.load("data/latent_and_output_h4/entangle_example_outs.npy")

    x_label_list = ['Kick', 'Snare', 'Cl. HH', 'Op. HH', "L. Tom", "M. Tom", "H. Tom", "Ride", "Crash", "Vibra", "H. Bongo", "L. Bongo", "Brass Hit", "Sax Hit"]

    for i in range(len(outs)):

        fig, ax = plt.subplots(1,1)

        img = ax.imshow(np.round(outs[i].reshape(14,32) + 0.4))
        #img = ax.imshow(outs[i].reshape(14,32))

        ax.set_yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13])

        ax.set_yticklabels(x_label_list)

        ax.set_title(hiddens[i])

        plt.show()

if True:
    outs = np.zeros((4,448))

    for i in range(4):
        outs[i] = decoder.predict(hids[i].reshape(1,20))

    np.save("data/latent_and_output_h20/entangle_example_outs.npy",outs)

if True:
    outs = np.load("data/latent_and_output_h20/entangle_example_outs.npy")

    x_label_list = ['Kick', 'Snare', 'Cl. HH', 'Op. HH', "L. Tom", "M. Tom", "H. Tom", "Ride", "Crash", "Vibra", "H. Bongo", "L. Bongo", "Brass Hit", "Sax Hit"]

    for i in range(len(outs)):

        fig, ax = plt.subplots(1,1)

        img = ax.imshow(np.flip(np.round(outs[i].reshape(14,32) + 0.4), axis = 1))
        #img = ax.imshow(outs[i].reshape(14,32))

        ax.set_yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13])

        ax.set_yticklabels(x_label_list)

        #ax.set_title(hiddens[i])

        plt.show()



