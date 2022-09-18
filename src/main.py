"""
This is the main program of the project.

This program lets the user interact with a bot. The bot
asks the user for a word (or a sentence), and then replies
with wether that word (or sentence) sounds more English or Swedish.

The bot gives sarcastical answers, and the level of sarcasm
depends on how certain the bot is of its answer.
"""

from tokenize import String
import tensorflow as tf
from tensorflow import keras as k
import numpy as np
import os
import random

def format_input(word: String) -> String:
    """
    Function that makes sure that the user input is correctly formatted.
    All words have to be less than 15 characters and only include valid lowercase characters.

    Returns:
    A correctly formatted string of the user input.
    """

    characters = "abcdefghijklmnopqrstuvwxyzåäö-é"
    out = ""
    added, i = 0, 0

    while added < 15:
        if added + i < len(word):
            if word[added + i].lower() in characters:
                out += word[added + i].lower()
                added += 1
            else:
                i += 1
        else:
            break
    return out


def layerize_word(word: String) -> np.ndarray:
    """Function that layerizes a word in order for the program to be able to feed it to the trained model"""

    char_dict = {"a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7, "i":8, "j":9, "k":10, "l":11, "m":12, "n":13, "o":14, "p":15, "q":16, "r":17, "s":18, "t":19, "u":20, "v":21, "w":22, "x":23, "y":24, "z":25, "å":26, "ä":27, "ö":28, "-":29, "é":30}
    layer = []

    for i in range(15):
        tmp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        if i < len(word):
            tmp[char_dict[word[i]]] = 1
        layer.append(tmp)
            
    return np.array(layer)


def start_message() -> None:
    """Procedure that prints a message to the user when the program starts """

    os.system("cls")
    print("--- WORD GUESSER ---")
    print("Enter a word, a sentence or a text and I will tell you if it sounds Swedish or English!")
    print()


def main() -> None:
    """main function of the program"""

    model = k.models.load_model("models/guesser.model")
    #model = k.models.load_model("ord/text_bools.model")
    loop = True
    verbose = True

    start_message()
    class1 = ["That's a tough one... but I guess it's {0}", "I'm not sure, but If you'd point a gun against my head, I'd say {0}", "I think it's {0}, but I might be mistaken", "I have been thinking about it for a while and, after careful consideration, my final answer is Swenglish - with a touch of {0}", "What is that!? {0}, maybe?", "That's difficult, man. Cut me some slack. I think it's {0}, though."]
    class2 = ["That's most likely {0}!", "I think that's {0}!", "Probably {0}!", "That's gotta be {0}, no?", "Well, I think that's {0}"]
    class3 = ["Definitely {0}!", "{0} 100%!", "Why are you even asking? Obviously it's {0}...", "Don't you have anything more difficult for me? It's {0}, bro", "Hmmm... the answer is 42! No, wait... wrong question... ah yes, this was the stupid question about language. Obviously it's {0}", "{0}. Final answer."]

    over1, over2, over3 = [], [], []
    english_hard = ["my", "is"]
    swedish_hard = ["jag", "heter"]

    while loop:
        inp = input("> ")

        if inp == "What is the meaning of life, the universe and everything?" or inp == "what is the meaning of life, the universe and everything?":
            print(42)

        words = inp.split(" ")
        inputs = np.array([layerize_word(format_input(x)) for x in words])
        predictions = model.predict([inputs])

        count = 0
        prob_english = 1
        prob_swedish = 1
        prob_total = 0

        for i, pred in enumerate(predictions):
            if words[i] in swedish_hard:
                prob_english *= 0.001
                prob_swedish *= 0.999
                count += 0
            elif words[i] in english_hard:
                prob_english *= 0.999
                prob_swedish *= 0.001
                count += 1
            else:
                count += np.argmax(pred)
                prob_english *= pred[1] # Räkna bara med om 60 eller över?
                prob_swedish *= pred[0]
    
        language = ""

        if prob_english > prob_swedish:
            if not verbose:
                print("English")
            language = "English"
            prob_total = prob_english / (prob_english + prob_swedish)
        else:  
            if not verbose:
                print("Swedish")
            language = "Swedish"
            prob_total = prob_swedish / (prob_english + prob_swedish)
        
        if not verbose:
            print("p_swedish: " + str(prob_swedish))
            print("p_english: " + str(prob_english))
            print("p_total: " + str(prob_total))
        
        else:
            line = ""

            if prob_total < 0.8:
                if len(class1) == 0:
                    class1 = list(tuple(over1))
                    over1 = []

                r = random.randint(0, len(class1) - 1)
                line = class1.pop(r)
                over1.append(line)

            elif prob_total < 0.95:
                if len(class2) == 0:
                    class2 = list(tuple(over2))
                    over2 = []

                r = random.randint(0, len(class2) - 1)
                line = class2.pop(r)
                over2.append(line)

            else:
                if len(class3) == 0:
                    class3 = list(tuple(over3))
                    over3 = []

                r = random.randint(0, len(class3) - 1)
                line = class3.pop(r)
                over3.append(line)

            print(line.format(language))
        print()

        # NOTES ON PERCENTAGES
        # ----------------------------
        # 0%-60% unknown
        # 60%-80% uncertain
        # 80%-95% certain
        # >95% very certain
        # ----------------------------


if __name__ == "__main__":
    main()