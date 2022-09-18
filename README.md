# natural-language-identifier

This is a bot that asks the user for a word (or a sentence), and then replies
with wether that word/sentence sounds more English or Swedish. The bot gives sarcastical answers, and the level of sarcasm
depends on how certain the bot is of its answer.

----------------

The project was created with TensorFlow using a FFN neural network.

The network has 1 layer consisting of 4096 nodes, and uses the optimizer "RMSprop" and the loss-function "sparse_categorical_crossentropy". It was trained on a data set consisting of 80 000 words for 10 epochs. It achieved 95% accuracy on a different dataset used for testing.

In the future, I will also implement a LSTM neural network. I will also add support for more languages than just Swedish and English.
