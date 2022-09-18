"""This code was used to preprocess the dictionaries into a trainable format for network.py"""

import random
from numpy import diff

def scramble(word_list: list[str]) -> list[str]:
    """Scrambles a list of words"""
    out = []
    length = len(word_list)

    for i in range(length):
        r = random.randint(0, length - i - 1)
        out.append(word_list[r])

        del word_list[r]
    return out


def get_words_raw() -> tuple[list[str]]:
    """Reads the raw data and outputs two lists (for the two languages) of scrambled words"""

    n_training = 40000
    n_test = 10000

    with open("data/swedish.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = list(map(lambda x: x[:-1].lower(), lines))
        lines_tmp = []

        for line in lines:
            if not " " in line and len(line) <= 15:
                lines_tmp.append(line)
        
        lines = lines_tmp

    tmp = scramble(lines)
    swedish_training = tmp[:n_training]
    swedish_test = tmp[n_training:n_training+n_test]

    with open("data/english.txt", "r", encoding="utf-8") as f:
        lines_tmp = f.readlines()
        lines_tmp = list(map(lambda x: x[:-1].lower(), lines_tmp))
        lines2 = []

        for line in lines_tmp:
            if len(line) <= 15:
                lines2.append(line)
        
    tmp2 = scramble(lines2)
    english_training = tmp2[:n_training]
    english_test = tmp2[n_training:n_training+n_test]

    swedish_training = [(i, 0) for i in swedish_training]
    swedish_test = [(i, 0) for i in swedish_test]

    english_training = [(i, 1) for i in english_training]
    english_test = [(i, 1) for i in english_test]

    out_training = swedish_training
    out_test = swedish_test

    out_training.extend(english_training)
    out_test.extend(english_test)

    return (scramble(out_training), scramble(out_test))


def get_words() -> tuple[list[str]]:
    """
    Procedure that reads the raw data and outputs 
    a tuple containing training words with their 
    corresponding labels for the two languages
    """

    training, testing = get_words_raw()
    training_words, training_labels = [], []
    testing_words, testing_labels = [], []

    for i in training:
        training_words.append(i[0])
        training_labels.append(i[1])

    for i in testing:
        testing_words.append(i[0])
        testing_labels.append(i[1])

    
    return (training_words, training_labels, testing_words, testing_labels)


def main() -> None:
    characters = "abcdefghijklmnopqrstuvwxyzåäö-é"
    training_words, training_labels, testing_words, testing_labels = get_words()


if __name__ == "__main__":
    main()