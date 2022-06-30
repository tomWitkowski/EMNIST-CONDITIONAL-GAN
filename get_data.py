import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf

trans_image = lambda im, lab: (tf.image.transpose(im),lab)
normalize = lambda im, lab: ( 1 - im/255, lab )


def add_spaces(images, labels):
    n = int(images.shape[0]/len(set(labels)))
    
    space_ims = tf.ones( [n]+list(images.shape[1:]) ) 
    space_lab = tf.repeat(max(labels)+1,n)
    
    images = tf.concat([images, space_ims],0)
    labels = tf.concat([labels, space_lab],0)
    
    return images, labels


def process_emnist_data(dataset, BATCH_SIZE):
    
    dataset = dataset.map(trans_image)
    dataset = dataset.map(normalize, num_parallel_calls = -1)
    dataset = dataset.cache()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(BATCH_SIZE)
    
    mdataset = np.concatenate([x for x, y in dataset], axis=0)
    number_dataset = np.concatenate([y for x, y in dataset], axis=0)
    
    return add_spaces(mdataset, number_dataset)


def get_emnist(dataset: str = 'balanced', batch_size: int = 32):

    (train,test), info = tfds.load(f'emnist/{dataset}', split= ['train', 'test'], shuffle_files = True, as_supervised=True, with_info=True)

    mtrain, number_train = process_emnist_data(train, batch_size)
    mtest, number_test = process_emnist_data(test, batch_size)

    return mtrain, number_train, mtest, number_test
    
