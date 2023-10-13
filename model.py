import os
import numpy as np
import pickle
import json
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model

from utils import get_config_from_path
from experiment import CodApSimulator

def load_dataset(config_path, n_masks):
    # Load the config file
    config = get_config_from_path(config_path)
    config_name = config_path.split('/')[-1].split('.json')[0]
    X = []
    Y = []
    for i in range(n_masks):
        # Modify the "mask_type" of the "source" to the current matrix
        config["source"]["mask_type"] = f"training_sources_{i}.txt"
        config["options"]["name"] = f"{config['options']['name'][:-2]}_{i}"
        # Get the data from the simulator.pkl object
        simulator = pickle.load(open(f"results/{config_name}/{config['options']['name']}/simulator.pkl", "rb"))
        x = simulator.sensor.screen
        y = simulator.source.screen
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)

    # Renormalize the data
    X = X / np.max(X)
    Y = Y / np.max(Y)
    return X, Y

def build_autoencoder(input_img_shape, output_img_shape, code_size=200):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(input_img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(output_img_shape))) # np.prod(output_img_shape) is the same as 32*32*3, it's more generic than saying 3072
    decoder.add(Reshape(output_img_shape))

    return encoder, decoder

if __name__ == '__main__':
    config_path = 'configs/autoencoder_experiment_2.json'
    config_name = config_path.split('/')[-1].split('.json')[0]
    X, Y = load_dataset(config_path=config_path, n_masks=90)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # Same as (32,32,3), we neglect the number of instances from shape
    INPUT_IMG_SHAPE = X.shape[1:]
    OUTPUT_IMG_SHAPE = Y.shape[1:]
    encoder, decoder = build_autoencoder(INPUT_IMG_SHAPE, OUTPUT_IMG_SHAPE, 100)

    inp = Input(INPUT_IMG_SHAPE)
    code = encoder(inp)
    reconstruction = decoder(code)

    autoencoder = Model(inp,reconstruction)
    autoencoder.compile(optimizer='adamax', loss='mse')

    print(autoencoder.summary())

    history = autoencoder.fit(x=X_train, y=Y_train, epochs=50,
                validation_data=[X_test, Y_test])

    # Save model
    autoencoder.save(f"models/{config_name}/autoencoder.h5")
    # Copy the config file to the models/{config_name} folder
    os.system(f"cp {config_path} models/{config_name}/config.json")

    # predict the autoencoder output from test data
    Y_pred = autoencoder.predict(X_test)
    # Visualize the first 5 test input and output pairs
    fig, axes = plt.subplots(3, 1, figsize=(20, 4))
    im0 = axes[0].imshow(X_test[0])
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(Y_pred[0])
    plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(Y_test[0])
    plt.colorbar(im2, ax=axes[2])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()