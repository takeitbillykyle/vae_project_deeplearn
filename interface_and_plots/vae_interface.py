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


nice_init_20 = np.array([[-0.31388889, -0.85833333, -0.83333333, -0.9,        -0.99166667, -0.41666667,
   0.31388889,  0.95277778, 0.475,      -0.97222222,         -0.28888889,  0.74166667,
   0.51111111, -0.075,      -0.46944444, -0.29166667, -0.63055556,  0.25277778,
   0.75277778 ,-0.375     ]])

arc = 20

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

def main(network = vae):

    #pygame variables
    matrix_element_unit = 10
    done = False
    clock = pygame.time.Clock()
    size = (720 + 32*matrix_element_unit, 720)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Drum Manifold Explorer")
    pattern_size = 3
    WHITE = (255,255,255)
    BLACK = (0,0,0)

    control_dots = int(arc / 2)

    marker_positions = np.zeros((10,2))

    #midi playback variables
    midi_player =  midiBuffer(device=[], verbose=True)
    bpm = 160
    beat_resolution = 4 #16 notes
    updates_per_second = (bpm / 60) * beat_resolution # bps/
    t = 0

    #network variables
    x_scale = 2/720
    x_shift = -1
    y_scale = 2/720
    y_shift = -1
    hidden_activations = np.zeros((1,arc))
    if arc == 20:
        hidden_activations = nice_init_20
        marker_positions = ((nice_init_20 - x_shift)/x_scale).reshape(10,2)
    output_activations = np.zeros((14,32))
    pattern = np.zeros((14,32))
    pygame.font.init()
    font_string = pygame.font.get_default_font()
    font = pygame.font.Font(font_string, 10)
    font2 = pygame.font.Font(font_string, 20)
    texts = [None]*control_dots
    origin_label = font.render("origin", True, BLACK, WHITE)
    output_activations_label = font2.render("output neuron activations", True, BLACK, WHITE)
    output_filtered_label = font2.render("drum pattern", True, BLACK, WHITE)
    
    for i in range(control_dots):
        texts[i] = font.render(str(i), True, BLACK, WHITE)


    output_activations = decoder.predict(hidden_activations)
    output_activations = np.flip(output_activations[0].reshape(14,32), axis = 1)

    cut_offs = np.load("data/optimal_cutoffs_h" + str(arc) + ".npy").reshape(14,32)

    if False:
        pattern = np.round(output_activations + (0.5 - cut_offs))
    else:  
        pattern = np.round(output_activations + 0.4)
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

                    hidden_activations[0,pressed_number*2] = x_val
                    hidden_activations[0,pressed_number*2 + 1] = y_val
                    print("xval")
                    print(x_val)
                    print("yval")
                    print(y_val)
                    marker_positions[pressed_number,0] = pos[0]  
                    marker_positions[pressed_number,1] = pos[1]
                    
                    print("pressed a " + sr(pressed_number)) 

                except:
                    print("press a number keys")

                print("hidden_activations")
                print(hidden_activations)

                output_activations = decoder.predict(hidden_activations)
               
                output_activations = np.flip(output_activations[0].reshape(14,32),axis = 1)
                
                pattern = np.round(output_activations + 0.4)

                #print("activations")
                #print(output_activations)

            if event.type == pygame.QUIT:
                done = True 

        #Init draw surface for interface
        screen.fill(WHITE) 
        
        #Draw interface here
        pygame.draw.rect(screen, (0,255,0), [360,360,pattern_size,pattern_size])


        for i in range(0,720,10):
            pygame.draw.line(screen, (200,200,200), (i,0), (i,720), 1)
            pygame.draw.line(screen, (200,200,200), (0,i), (720,i), 1)

        pygame.draw.line(screen, (200,200,200), (0,360), (720,360), 3)
        pygame.draw.line(screen, (200,200,200), (360,0), (360,720), 3)
 
        for i in range(control_dots):
            color = (255*(control_dots - i)/control_dots,0,255*i/control_dots)
            pygame.draw.rect(screen, color, [marker_positions[i,0],marker_positions[i,1],pattern_size,pattern_size])
            
            text_rec = texts[i].get_rect()
            text_rec.center = (marker_positions[i,0] + 5, marker_positions[i,1])
            screen.blit(texts[i], text_rec)

        for i in range(14):
            for j in range(32):
                node_activation = output_activations[i,j]
                pattern_activation = pattern[i,j]
                pygame.draw.rect(screen, (150,200*(1 - node_activation),200*(1 - node_activation)), 
                    [720 + j*matrix_element_unit, i*matrix_element_unit, matrix_element_unit,matrix_element_unit])
                
                pygame.draw.rect(screen, (255*(1 - pattern_activation), 200*(1 - pattern_activation),200), 
                    [720 + j*matrix_element_unit, i*matrix_element_unit + matrix_element_unit * 32, matrix_element_unit,matrix_element_unit])
        
        text_rec_act = output_activations_label.get_rect()
        text_rec_act.center = (720 + int(320/2), matrix_element_unit * 16)

        text_rec_filt = output_filtered_label.get_rect()
        text_rec_filt.center = (720 + int(320/2), matrix_element_unit * (16 + 32))

        screen.blit(output_activations_label, text_rec_act)
        screen.blit(output_filtered_label, text_rec_filt)
        #Drawing is displayed on screen through flip() method
        pygame.display.flip()

        #Play midi

        for instr in range(len(drum_map)):
            if t%1 == 0:
                if pattern[instr,int(t)] > 0:
                    midi_player.playChord([int(drum_map[instr])], 500, int(pattern[instr, int(t)]*127))

        t += 1
        t %= 32

        #Limit framerate 
        time.sleep((updates_per_second**-1))

    pygame.quit()

if __name__ == "__main__":
    main()
