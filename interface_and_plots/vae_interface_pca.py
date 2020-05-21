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
import pygame
import time
from midiConvert import drum_map
from midiBuffer import midiBuffer

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
epochs = 100
batch_size = 64
vae.fit(x_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test, None))

vae.save_weights('vae_100e_64b.h5')
'''
vae.load_weights(arc_name)

eigen_vectors = np.load("data/latent_pca_comps.npy")
latent_activations = np.zeros((6,5000,20))


for i in range(6):
    latent = np.load("data/latent_matrix_" + str(i) + ".npy")
    print(latent.shape)
    latent_activations[i] = latent

latent_activations = np.array(latent_activations)

latent = np.concatenate(latent_activations, axis = 0)
latent_pca = np.dot(latent,eigen_vectors.T)

def main(network = vae):

    #pygame variables
    matrix_element_unit = 10
    done = False
    clock = pygame.time.Clock()
    size = (720 + 32*matrix_element_unit, 720)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Drum Manifold Explorer")
    pattern_size = 10
    WHITE = (255,255,255)
    BLACK = (0,0,0)
    
    pca_dims = 6
    control_dots = int(pca_dims / 2)
    pca_activations = np.zeros((pca_dims))

    eigen_vectors = np.load("data/latent_pca_comps.npy")
    eigen_vectors = eigen_vectors[:pca_dims,:]

    train_hidden_coords = latent_pca[:,:pca_dims]

    screen_div_length = int(1200 / control_dots)

    #midi playback variables
    midi_player =  midiBuffer(device=[], verbose=True)
    bpm = 120
    beat_resolution = 4 #16 notes
    updates_per_second = (bpm / 60) * beat_resolution # bps/
    t = 0

    #network variables
    x_scale = 2/720
    x_shift = -1
    y_scale = 2/720
    y_shift = -1
    hidden_activations = np.zeros((1,20))
    output_activations = np.zeros((14,32))
    pattern = np.zeros((14,32))

    marker_positions = np.zeros((10,2)) + x_shift

    #interface loop
    while not done:

        for event in pygame.event.get():

            if event.type == pygame.KEYDOWN: #pos[0] is x, pos[1] is y
                event_dict = event.__dict__ # {'pos': (x, y), 'button': 1, 'window': None})
                pos = pygame.mouse.get_pos()
                
                x_val = pos[0]*x_scale + x_shift
                y_val = pos[1]*y_scale + y_shift

                try:
                    print("unicode is")
                    print(int(event_dict['unicode']))
                    pressed_number = int(event_dict['unicode'])

                    pca_activations[pressed_number*2] = x_val
                    pca_activations[pressed_number*2 + 1] = y_val
                    print("xval")
                    print(x_val)
                    print("yval")
                    print(y_val)
                    marker_positions[pressed_number,0] = pos[0]  
                    marker_positions[pressed_number,1] = pos[1]
                    
                    print("pressed a " + str(pressed_number)) 
                
                except:
                    print("press a number in range " + str(control_dots))
                
                hidden_activations = (pca_activations.T @ eigen_vectors).reshape(1,20)

                print("hidden_activations")
                print(hidden_activations)

                output_activations = decoder.predict(hidden_activations)
               
                output_activations = output_activations[0].reshape(14,32)
                
                pattern = np.round(output_activations + 0.4)

                #print("activations")
                #print(output_activations)

            if event.type == pygame.QUIT:
                done = True 

        #Init draw surface for interface
        screen.fill(WHITE) 
        
        #Draw interface here
        for i in range(control_dots):
            color = (255*(control_dots - i)/control_dots,0,255*i/control_dots)
            color2 = (125*(control_dots - i)/control_dots,125,125*i/control_dots)
            '''
            #roll activations and pca projection coordinates
            this_plane = np.roll(train_hidden_coords, -i*2, axis = 1)
            this_plane_activation = np.round(np.roll(pca_activations, -i*2), decimals = 2)

            coord_set = []

            #find all points that match set coordinates in other dims
            temp_plane = np.roll(this_plane,-2, axis = 1)
            for j in range(pca_dims - 2):
                check_dim = np.round(this_plane[:,2 + j], decimals = 2)
                print("activation in node " + str(j))
                print( this_plane_activation[2 + j])
                coord_set = np.where( check_dim == temp_plane[:,0])[0]
                print("number of matching")
                print(coord_set.shape[0])
                temp_plane = temp_plane[coord_set]
                print("temp plane shape")
                print(temp_plane.shape)
                #temp_plane = np.roll(temp_plane, -1, axis = 1)

            '''
            screen_div_length = 0
            pygame.draw.rect(screen, color, [marker_positions[i,0] + i*screen_div_length,marker_positions[i,1],pattern_size,pattern_size])
 
        for i in range(14):
            for j in range(32):
                node_activation = output_activations[i,j]
                pygame.draw.rect(screen, (255*node_activation, 0,0), 
                    [720 + j*matrix_element_unit, i*matrix_element_unit, matrix_element_unit,matrix_element_unit])

        #Drawing is displayed on screen through flip() method
        pygame.display.flip()

        #Play midi

        for instr in range(len(drum_map)):
            if t%2 == 0:
                if pattern[instr,int(t/2)] > 0:
                    midi_player.playChord([int(drum_map[instr])], 500, int(pattern[instr, int(t/2)]*127))

        t += 1
        t %= 64

        #Limit framerate 
        time.sleep((updates_per_second**-1)/2)

    pygame.quit()

if __name__ == "__main__":
    main()
