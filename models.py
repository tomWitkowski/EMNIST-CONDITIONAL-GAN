import tensorflow as tf
from tqdm import tqdm
import numpy as np
from cfg import *

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
try:
    for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
except: 
    ...

# tf.debugging.set_log_device_placement(True)

class Generator(tf.keras.Model):
    """
    Generator transforming random noise and letter/number into EMNIST-style image
    """
    @staticmethod
    def gray_rgb(im):
        im = tf.reduce_mean(im,-1, keepdims=True)
        return tf.concat([im]*3,-1)

    def __init__(self):
        noise = layer = tf.keras.Input(shape=(LATENT_DIM,), name='random_noise')
        N = number = tf.keras.Input(shape=(), name='number_label', dtype = tf.int32)

        # 3 - > [0,0,0,1,0,0,0,0,0,0,0]
        number = tf.one_hot(number, N_CLASS, axis=-1)

        layer = tf.concat([layer, number], 1)

        layer = tf.keras.layers.Dense(7*7*16)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.LeakyReLU(0.1)(layer)
        
        layer = tf.reshape(layer,(-1,7,7,16))
        
        layer = tf.keras.layers.Conv2D(16, (3,3) , padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.LeakyReLU(0.1)(layer)
        layer = tf.keras.layers.Dropout(0.1)(layer)
        
        layer = tf.keras.layers.UpSampling2D( (2,2) )(layer)
        layer = tf.keras.layers.Conv2D(32, (4,4) , padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.LeakyReLU(0.1)(layer)
        layer = tf.keras.layers.Dropout(0.1)(layer)

        layer = tf.keras.layers.UpSampling2D( (2,2) )(layer)
        layer = tf.keras.layers.Conv2D(16, (5,5) , padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.LeakyReLU(0.1)(layer)
        layer = tf.keras.layers.Dropout(0.1)(layer)
            
        layer = tf.keras.layers.Conv2D(SHAPE[-1], (5,5) , padding='same')(layer)

        layer = (tf.keras.activations.tanh(layer)+1)/2
        layer = 1 - layer # black - one, white - zero

        # layer = self.gray_rgb(layer)
        output_layer = layer

        super().__init__(inputs = [noise, N], outputs = [output_layer])

    def comp(self):
        self.compile(optimizer='adam', 
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

    
class Discriminator(tf.keras.Model):
    """
    Classifier predicting:
        > Which letter/number is an image
        > Is a letter/number on image at all
    """
    @staticmethod
    def dense(layer: object, neurons: int) -> object:
        layer = tf.keras.layers.Dense(neurons)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.LeakyReLU(0.1)(layer)
        layer = tf.keras.layers.Dropout(0.1)(layer)
        return layer

    @staticmethod
    def scaled_tanh(layer: object) -> object:
      """
      tanh transformation ((-1,1)+1)/2 = (0,1)
      """
      return (tf.keras.activations.tanh(layer)+1)/2

    def __init__(self):
        input_layer = layer = tf.keras.Input(shape=SHAPE, name='image')
        
        layer = tf.keras.layers.Conv2D(32, (6,6), strides=(2,2), padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.LeakyReLU(0.1)(layer)
        layer = tf.keras.layers.Dropout(0.1)(layer)

        layer = tf.keras.layers.Conv2D(64, (5,5), strides=(2,2), padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.LeakyReLU(0.1)(layer)
        layer = tf.keras.layers.Dropout(0.1)(layer)

        layer = tf.keras.layers.Conv2D(88, (4,4), strides=(2,2), padding='same')(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.LeakyReLU(0.1)(layer)
        layer = tf.keras.layers.Dropout(0.1)(layer)
            
        layer = tf.keras.layers.Flatten()(layer)

        layer = self.dense(layer, 60)
        layer = self.dense(layer, 50)
        
        # predict whether the image is real or fake
        is_real = self.dense(layer, 30)
        is_real = tf.keras.layers.Dense(10)(is_real)
        is_real = tf.keras.layers.Dense(1)(is_real)
        is_real = self.scaled_tanh(is_real)

        # classify
        what = tf.keras.layers.Dense(N_CLASS)(layer)
        what = tf.keras.activations.softmax(what)

        output_layer = is_real * what
        
        super().__init__(inputs = [input_layer], outputs = [output_layer])

    def comp(self):
        self.compile(optimizer='adam', 
                    loss = 'categorical_crossentropy',
                    metrics = ['accuracy'])
    
    
class GAN:
    """
    Conditional GAN for generating specific text
    """
    def __init__(self):

        self.gen_acc = []
        self.dis_valid_acc = []
        self.dis_fake_acc = []

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.discriminator.comp()

        self.discriminator.trainable = False
        
        self.gan_noise_input = tf.keras.Input(shape=(LATENT_DIM,))
        self.gan_number_input = tf.keras.Input(shape=(), dtype = tf.int32)

        self.gan_input = [self.gan_noise_input, self.gan_number_input]
        self.gan_output = self.discriminator(self.generator(self.gan_input))

        self.model = tf.keras.Model(self.gan_input, self.gan_output) 
        self.model.compile('adam', 'categorical_crossentropy', metrics = ['accuracy'])

        
    def __call__(self, x):
        return self.model(x)
      
        
    @staticmethod
    def mix_generated(generated, batch_size: int):
        
        mixed = tf.random.shuffle(generated)[:int(batch_size/4)]
        mixed2 = tf.random.shuffle(generated)[:int(batch_size/4)]
        
        mixedv = tf.concat([mixed2[:,14:,:,:], mixed[:,:14,:,:]],1)
        mixedh = tf.concat([mixed2[:,:,14:,:], mixed[:,:,:14,:]],2)
        
        return tf.concat([mixedv, mixedh],0)
        
        
    def train_discriminator(self, images: object, labels: list, batch_size: int, mix_fit: bool = False, add_mixed: bool = True):

        self.discriminator.trainable = True

        # # Train on true images
        indices = np.random.randint(0, images.shape[0], batch_size)
        true_images = images[indices]
        true_number = labels[indices]

        valid = tf.one_hot(true_number, N_CLASS)
        fake = np.zeros((batch_size, N_CLASS))

        # # Train on generated images
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))
        fake_images = self.generator([noise, true_number])

        
        if add_mixed:
            fake_images = tf.concat([fake_images, self.mix_generated(fake_images, batch_size)], 0)
            fake = tf.zeros((fake_images.shape[0], N_CLASS))
                   
        if mix_fit:
            mix_input = tf.concat([true_images, fake_images], 0)
            mix_output = tf.concat([valid, fake], 0)
            tmp = self.discriminator.train_on_batch(mix_input, mix_output)[1]
            self.dis_valid_acc.append( tmp )
            self.dis_fake_acc.append( tmp )

        else:
            self.dis_valid_acc.append( self.discriminator.train_on_batch(true_images, valid)[1] )
            self.dis_fake_acc.append( self.discriminator.train_on_batch(fake_images, fake)[1] )

        self.discriminator.trainable = False

        
    def train_generator(self, labels, batch_size: int):

        indices = np.random.randint(0, len(labels), batch_size)
        true_number = labels[indices]
        valid = tf.one_hot(true_number, N_CLASS)

        noise = np.random.normal(0, 1, (batch_size,LATENT_DIM))

        self.gen_acc.append( self.model.train_on_batch([noise, true_number], valid)[1] )


    def train(self, images: object, labels: object, epochs: int, batch_size: int, mix_fit: bool = False, revert_original: bool = True, add_mixed: bool = True):
        """
        Parameters
        ----------
        
        images : some set e.g. np.ndarray
            set of images 
        labels : some set e.g. np.ndarray
            letters/numbers
        epochs: int
            number of iteration with gradient computing
        batch_size : int
            number of processed images in one iteration
        mix_fit : bool
            whether to mix fake and real images inside batch
        revert_original: bool
            whether to mirror the image
        add_mixed: bool
            whether to add parts of letters as a fake - thank to this discriminator classifies better
        """
        for e in tqdm(range(epochs)):
            if revert_original:
                self.train_discriminator(images if e%2 else images[:,:,::-1,:], labels, batch_size, mix_fit, add_mixed)
            else:
                self.train_discriminator(images, labels, batch_size, mix_fit, add_mixed)

            self.train_generator(labels, batch_size)

            
    def load_weights(self, directory: str = 'weights', 
                     generator_weights_name: str = 'generator', 
                     discriminator_weights_name: str = 'discriminator'):
        """
        Load saved weights from directory
        """
        generator_weights = np.load(f'{directory}/{generator_weights_name}.npy', allow_pickle=True)
        discriminator_weights = np.load(f'{directory}/{discriminator_weights_name}.npy', allow_pickle=True)
        
        self.generator.set_weights(generator_weights)
        self.discriminator.set_weights(discriminator_weights)
        
        
    def generate_row(self, row: list, rand: np.ndarray):
        generated = self.generator([rand, row])
        generated = tf.squeeze(tf.concat(tf.split(generated, generated.shape[0], axis=0), 2), axis=0)

        rm = tf.reduce_mean(generated,-1, keepdims=True)
        return tf.concat([rm]*3,-1)
    
    
    def generate_on_table(self, table: list, rand: np.ndarray):
        return tf.concat([self.generate_row(row, rand) for row in table],0).numpy() 