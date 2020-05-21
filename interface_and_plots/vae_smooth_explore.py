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
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

arc = 20

if arc == 4:
    arc_name = "4_latent_512_256_100e_64b.h5"
    root = "data/latent_and_output_h4/"
if arc == 20:
    arc_name = "20_latent_256_128_100e_64b.h5"
    root = "data/latent_and_output_h20/"

if False:
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

num_trajectories = 50
num_steps = 1000
step_length = 0.001

trajectories = np.random.rand(num_trajectories,1,arc)*2 - 1
old_output_activation = [0]*num_trajectories
output_activation = [0]*num_trajectories

if False:
    for i in range(num_trajectories):
        old_output_activation[i] = decoder.predict(trajectories[i])

    out_step_lengths = np.zeros((num_trajectories,num_steps))

    for i in range(num_steps):
        print(i)
        for j in range(num_trajectories):
            step = np.random.rand(1,arc)
            trajectories[j] += (step / np.linalg.norm(step)) * step_length
        
            output_activation[j] = decoder.predict(trajectories[j])

        #print(output_activation.shape())
        #input()
        for j in range(num_trajectories):
            out_step_lengths[j,i] = np.linalg.norm(output_activation[j] - old_output_activation[j])

        old_output_activation = output_activation.copy()

    np.save("data/latent_and_output_h" + str(arc) + "/out_step_lengths" + str(arc) + ".npy", out_step_lengths)

    print(out_step_lengths[0])

    for i in range(3):
        plt.plot(out_step_lengths[i])
    plt.show()

if True:
    step_lengths = np.load("data/latent_and_output_h" + str(arc) + "/out_step_lengths" + str(arc) + ".npy")

    for i in range(num_trajectories):
        mean = np.mean(step_lengths[i])
        std = np.std(step_lengths[i])

        if i == 0:
            plt.scatter(i,mean, color = 'g', label = "mean", marker = ".")
            plt.scatter(i,mean + std, marker = '1', color = 'b', label = "stand. dev.")
            plt.scatter(i,mean - std, marker = '2', color = 'b')
            plt.scatter(i, np.max(step_lengths[i],), marker = '*', color = 'r', label = "maximum") 
            plt.scatter(i, np.min(step_lengths[i],), marker = '*', color = 'c', label = "minimimum") 
        else:
            plt.scatter(i,mean, color = 'g', marker = ".")
            plt.scatter(i,mean + std, marker = '1', color = 'b')
            plt.scatter(i,mean - std, marker = '2', color = 'b')
            plt.scatter(i, np.max(step_lengths[i],), marker = '*', color = 'r') 
            plt.scatter(i, np.min(step_lengths[i],), marker = '*', color = 'c')

    mean = np.mean(step_lengths, axis = 1)
    std = np.std(step_lengths, axis = 1)
    maxim = np.max(step_lengths, axis = 1)
    minim = np.min(step_lengths, axis = 1)

    plt.plot(range(0,50),mean, color = 'g', alpha = 0.4)
    plt.plot(range(0,50),mean + std, color = 'b', alpha = 0.4)
    plt.plot(range(0,50),mean - std, color = 'b', alpha = 0.2)
    plt.plot(range(0,50),maxim, color = 'r', alpha = 0.4)
    plt.plot(range(0,50),minim, color = 'c', alpha = 0.3)
    
    plt.xlabel('Trajectory')
    plt.ylabel('Step length in output')
    plt.ylim([0,0.009])
    plt.legend()
    plt.show()
