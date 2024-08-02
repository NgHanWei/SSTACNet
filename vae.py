import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import constraints
from preprocess import get_data
import matplotlib.pyplot as plt
tf.keras.utils.set_random_seed(1104)

def Conv_block(input_layer, F1=4, kernLength=64, poolSize=8, D=2, in_chans=22, dropout=0.1):
    """ Conv_block
    
        Notes
        -----
        This block is the same as EEGNet with SeparableConv2D replaced by Conv2D 
        The original code for this model is available at: https://github.com/vlawhern/arl-eegmodels
        See details at https://arxiv.org/abs/1611.08024
    """
    F2= F1*D
    block1 = layers.Conv2D(F1, (kernLength, 1), padding = 'same',data_format='channels_last',use_bias = False)(input_layer)
    block1 = layers.BatchNormalization(axis = -1)(block1)
    block2 = layers.DepthwiseConv2D((1, in_chans), use_bias = False, 
                                    depth_multiplier = D,
                                    data_format='channels_last',
                                    depthwise_constraint = constraints.max_norm(1.))(block1)
    block2 = layers.BatchNormalization(axis = -1)(block2)
    block2 = layers.Activation('elu')(block2)
    block2 = layers.AveragePooling2D((8,1),data_format='channels_last')(block2)
    block2 = layers.Dropout(dropout)(block2)
    block3 = layers.Conv2D(F2, (16, 1),
                            data_format='channels_last',
                            use_bias = False, padding = 'same')(block2)
    block3 = layers.BatchNormalization(axis = -1)(block3)
    block3 = layers.Activation('elu')(block3)
    
    block3 = layers.AveragePooling2D((poolSize,1),data_format='channels_last')(block3)
    block3 = layers.Dropout(dropout)(block3)
    return block3

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
latent_dim = 8

input_1 = keras.Input(shape=(1, 22, 1125))
encoder_inputs = layers.Permute((3,2,1))(input_1)

block1_start = Conv_block(input_layer = encoder_inputs, F1 = 16, D = 2, 
                    kernLength = 64, poolSize = 8,
                    in_chans = 22, dropout = 0.3)

print(block1_start)
print(encoder_inputs)

# print(encoder_inputs)
x = layers.Conv2D(16, 3, activation="relu", strides=(2,5), padding="same")(block1_start)
print(x)
x = layers.Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(8, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(input_1, [z_mean, z_log_var, z], name="encoder")
# print(encoder.summary())

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(11 * 225 * 8, activation="relu")(latent_inputs)
x = layers.Reshape((11, 225, 8))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=1, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=(2,5), padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="tanh", padding="same")(x)
decoder_outputs = layers.Permute((3,1,2))(decoder_outputs)
# print(decoder_outputs)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
# print(decoder.summary())

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss * 0.5
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# data_path = "D:/atcnet/bci2a_mat/"
# dataset_conf = { 'n_classes': 4, 'n_sub': 9, 'n_channels': 22, 'data_path': data_path,
#                 'isStandard': True, 'LOSO': False}

class cut_data():
    def cut(dataset_conf,sub,LOSO,dataset):
        data_path = dataset_conf.get('data_path')
        LOSO = dataset_conf.get('LOSO')
        isStandard = dataset_conf.get('isStandard')
        X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(
            data_path, sub, dataset, LOSO = LOSO, isStandard = isStandard)
        # X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(
        #             data_path, sub, LOSO, isStandard)

        # print(X_train.shape)

        # (x_train, y_train), (x_test, _) = keras.datasets.mnist.load_data()
        # print(y_train)
        x_train = X_train
        x_test = X_test
        # mnist_digits = np.concatenate([x_train, x_test], axis=0)
        mnist_digits = X_train
        mnist_digits = mnist_digits.astype("float32")

        vae = VAE(encoder, decoder)
        vae.compile(optimizer=keras.optimizers.Adam())
        vae.fit(mnist_digits, epochs=100, batch_size=10)

        recon_loss_array = []
        recon_aug_array = []
        recon_loss_train = 0
        recon_loss_test = 0
        for i in range(0,len(x_train)):
            
            # plt.imshow(np.array(x_train[i][0]), cmap='gray')
            # plt.show()

            _, _, z = vae.encoder.predict(x_train[i:i+1])
            # print(z)
            recon = vae.decoder.predict(z)

            # plt.imshow(np.array(recon[0][0]), cmap='gray')
            # plt.show()

            reconstruction_loss = tf.reduce_mean(
                            tf.reduce_sum(
                                keras.losses.mean_squared_error(x_train[i:i+1], recon), axis=(1, 2)
                            )
            )
            recon_loss_train += reconstruction_loss

            ### Evaluation loss
            # _, _, z = vae.encoder.predict(x_test[i:i+1])
            # # print(z)
            # recon = vae.decoder.predict(z)

            # # plt.imshow(np.array(recon[0][0]), cmap='gray')
            # # plt.show()

            # reconstruction_loss = tf.reduce_mean(
            #                 tf.reduce_sum(
            #                     keras.losses.mean_squared_error(x_test[i:i+1], recon), axis=(1, 2)
            #                 )
            # )
            # recon_loss_test += reconstruction_loss

            recon_loss_array.append(float(reconstruction_loss))
            if len(recon_aug_array) == 0:
                recon_aug_array = recon
            else:
                recon_aug_array = np.concatenate([recon_aug_array, recon+x_train[i:i+1]], axis=0)

        # print(recon_loss_array)
        order_list = [i for i in range(len(x_train))]
        list1, order_list = (list(x) for x in zip(*sorted(zip(recon_loss_array, order_list))))
        # print(order_list)
        if LOSO == False:
            cut_off_list = order_list[:280]
        else:
            cut_off_list = order_list[:460]
        ## Data reduction
        # new_x_train = x_train[cut_off_list]
        # new_y_train = y_train_onehot[cut_off_list]
        # print(new_x_train.shape)

        ## Data augmentation
        new_x_train = np.concatenate([x_train,recon_aug_array[cut_off_list]],axis=0)
        new_y_train = np.concatenate([y_train_onehot,y_train_onehot[cut_off_list]],axis=0)
        # print(recon.shape)

        print(recon_loss_train)
        print(recon_loss_test)

        return new_x_train, new_y_train

# import matplotlib.pyplot as plt
# def plot_label_clusters(vae, data, labels):
#     # display a 2D plot of the digit classes in the latent space
#     z_mean, _, _ = vae.encoder.predict(data)
#     print(z_mean.shape)
#     # plt.figure(figsize=(12, 10))
#     plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
#     plt.colorbar()
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.show()

# y_train = y_train_onehot
# test_y = np.where(y_train==1)[1]
# plot_label_clusters(vae, x_train, test_y)

# data_path = "D:/atcnet/bci2a_mat/"
# dataset_conf = { 'n_classes': 4, 'n_sub': 9, 'n_channels': 22, 'data_path': data_path,
#                 'isStandard': True, 'LOSO': False}
# sub = 3
# X_train,y_train_onehot = cut_data.cut(dataset_conf=dataset_conf,sub=sub)