"""This is the code that trains the neural network using the provided data.

Characteristics of the network:
Type: FFN
Layers: 1
Nodes: 4096
Optimizer: RMSprop
Loss function: sparse_categorical_crossentropy
Dataset size: 40 000 words
Epochs: 10
"""

import preprocessing
import tensorflow as tf
from tensorflow import keras as k
import numpy as np

def layerize(word_list: list[str], use_bools: bool = True) -> np.ndarray:
    """Function that layerizes a word in order for the program to be able to feed it to the trained model"""

    char_dict = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7, "i":8, "j":9, "k":10, "l":11, "m":12, "n":13, "o":14, "p":15, "q":16, "r":17, "s":18, "t":19, "u":20, "v":21, "w":22, "x":23, "y":24, "z":25, "å":26, "ä":27, "ö":28, "-":29, "é":30}
    out = []

    if use_bools:
        for word in word_list:
            layer = []

            for i in range(15):
                tmp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

                if i < len(word):
                    tmp[char_dict[word[i]]] = 1

                layer.append(tmp)         
            out.append(np.array(layer))

    else:
        for word in word_list:
            layer = []

            for i in range(15):
                tmp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

                if i < len(word):
                    tmp[char_dict[word[i]]] = 1
                    layer.append((char_dict[word[i]] + 1) / 31) ##
                else:
                    layer.append(0)              

            out.append(np.array(layer))
    return np.array(out)


def main() -> None:
    """Procedure that trains the neural network"""

    use_bools = True
    training_words_tmp, training_labels, testing_words_tmp, testing_labels = preprocessing.get_words()

    training_words, testing_words = layerize(training_words_tmp, use_bools=use_bools), layerize(testing_words_tmp, use_bools=use_bools)
    training_labels, testing_labels = np.array(training_labels), np.array(testing_labels)

    # new_model = k.models.load_model("words1.model")
    model = k.models.Sequential()

    if use_bools:
        model.add(k.layers.Flatten(input_shape=(15, 31)))
        model.add(k.layers.Dense(4096, activation=tf.nn.relu))
        model.add(k.layers.Dense(2, activation=tf.nn.softmax))

        model.compile(optimizer="RMSprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(training_words, training_labels, epochs=10) # was 20

        model.save("models/guesser2.model") # were no 2
        # model.save("models/text_bools.model")

    else:

        model.add(k.layers.Flatten(input_shape=(15, 1))) # (15, 31)
        model.add(k.layers.Dense(4096, activation=tf.nn.relu))
        model.add(k.layers.Dense(4096, activation=tf.nn.relu))
        model.add(k.layers.Dense(2, activation=tf.nn.softmax))

        model.compile(optimizer="nadam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(training_words, training_labels, epochs=20)
        model.save("models/text_ints.model")


    # NOTES ON MODELLING
    # --------------------------------------------------------------
    # 1 for 8192, epoch=15, RMSprop * try longer? alos try 40k
    # 1 for 4096, epoch=10, RMSprop *
    # 1 for 2048, epoch=10, adamax
    # 1 for 2048, epoch=5, adam
    # 1 for 2048, epoch=10, nadam? try slightly longer
    # 1 for 1024, epoch=5
    # 1 for 128, epoch=5
    # 2 for 256, epoch=5
    # 1 for 64, epoch=5
    # --------------------------------------------------------------

    val_loss, val_acc = model.evaluate(testing_words, testing_labels)
    predictions = model.predict([testing_words])
    pred = np.argmax(predictions[32])

    print()
    print("Word: " + str(testing_words_tmp[32]))
    print("Prediction: " + str(pred))
    print("Actual: " + str(testing_labels[32]))
    print()


if __name__ == "__main__":
    main()