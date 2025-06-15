import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, -1)  
x_test = np.expand_dims(x_test, -1)

def create_pairs(x, y, num_pairs_per_class=5000):
    pairs = []
    labels = []
    num_classes = 10
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]
    rng = np.random.default_rng(42)
    for idx in range(num_classes):
        # Похожие пары
        idx1 = rng.choice(digit_indices[idx], num_pairs_per_class)
        idx2 = rng.choice(digit_indices[idx], num_pairs_per_class)
        for i, j in zip(idx1, idx2):
            pairs += [[x[i], x[j]]]
            labels += [1]
        # Разные пары
        for _ in range(num_pairs_per_class):
            inc = (idx + rng.integers(1, num_classes)) % num_classes
            i = rng.choice(digit_indices[idx])
            j = rng.choice(digit_indices[inc])
            pairs += [[x[i], x[j]]]
            labels += [0]
    return np.array(pairs), np.array(labels).astype('float32')
pairs_train, labels_train = create_pairs(x_train, y_train, num_pairs_per_class=2000)
pairs_test, labels_test = create_pairs(x_test, y_test, num_pairs_per_class=300)

def encoder(input_shape,latent_dim):
    inputs=keras.Input(shape=input_shape)
    conv3=layers.Conv2D(64,(3,3),padding='same',activation='relu')(inputs)

    conv5=layers.Conv2D(64,(5,5),padding='same',activation='relu')(inputs)
    conv7=layers.Conv2D(64,(7,7),padding='same',activation='relu')(inputs)
    merged=layers.concatenate([conv3,conv5,conv7])
    x=layers.MaxPooling2D(pool_size=(2,2))(merged)
    x=layers.Flatten()(x)
    latent=layers.Dense(latent_dim,name='latent',activation='softmax')(x)
    return keras.Model(inputs,latent,name='encoder')

input_shape = (28, 28, 1)  
latent_dim = 32
encode=encoder(input_shape,latent_dim)

input_a=keras.Input(shape=input_shape)
input_b=keras.Input(shape=input_shape)

embedding_a=encode(input_a)
embedding_b=encode(input_b)

distance=layers.Lambda(lambda x:tf.sqrt(tf.reduce_sum(tf.square(x[0]-x[1]),axis=1,keepdims=True))
                       )([embedding_a,embedding_b])
siamese_model = keras.Model(inputs=[input_a, input_b], outputs=distance)


def contrastive_loss(y_true,y_pred,margin=1.0):
    y_true=tf.cast(y_true,y_pred.dtype)
    return tf.reduce_mean(
        y_true*tf.square(y_pred)+(1-y_true)*tf.square(tf.maximum(margin-y_pred,0))
    )
siamese_model.compile(optimizer='adam', loss=contrastive_loss, metrics=[])
siamese_model.fit(
    [pairs_train[:, 0], pairs_train[:, 1]], labels_train,
    batch_size=128,
    epochs=5,
    validation_data=([pairs_test[:, 0], pairs_test[:, 1]], labels_test)
)
plt.figure(figsize=(8, 12))
for im in range(6):
    img1 = x_test[im]
    img2 = x_test[im+1]
    plt.subplot(6, 2, 2*im+1)
    plt.imshow(img1.squeeze(), cmap='gray')
    plt.title(f'Image {im}')
    plt.axis('off')
    plt.subplot(6, 2, 2*im+2)
    plt.imshow(img2.squeeze(), cmap='gray')
    plt.title(f'Image {im+1}')
    plt.axis('off')
    distance_val = siamese_model.predict([img1[np.newaxis], img2[np.newaxis]])
    print("Distance between images:", distance_val[0,0])
plt.show()

img4=x_test[4]
img6=x_test[6]
distance_val = siamese_model.predict([img4[np.newaxis], img6[np.newaxis]])
print("Distance between images:", distance_val[0,0])

