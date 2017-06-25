from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
import numpy as np
import random
import sys

data = open('female.txt')
names_file = open('female.txt').read().lower()
names = ""
names_list = []
for line in data:
	names += " " + ((line.split(None, 1)[0]).lower())
	names_list.append((line.split(None, 1)[0]).lower())


print('names list length:', len(names))
print('names file length:', len(names_file))
chars = set(names_file)

print('chars list length:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 30
step = 3
names = []
next_chars = []
for i in range(0, len(names_file)-maxlen, step):
	names.append(names_file[i: i + maxlen])
	next_chars.append(names_file[i + maxlen])
print('nb sequences:', len(names))

print("Vectorization...")
X = np.zeros((len(names), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(names), len(chars)), dtype=np.bool)
for i, name in enumerate(names_list):
	for t, char in enumerate(name):
		X[i, t, char_indices[char]] = 1
	y[i, char_indices[next_chars[i]]] = 1
print('input vector (X) shape:', X.shape)
print('output vector (y) shape:', y.shape)

#2 stacked LSTM
print("Building model..")
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

def generate_names(model, length):
	ix = [np.random.randint(len(names))]
	y_char = [indices_char[ix[-1]]]
	X = np.zeros((1, length, len(names)))
	for i in range(length):
		X[0, i, :][ix[-1]] = 1
		print(indices_char[ix[-1]], end="")
		ix = np.argmax(model.predict(X[:, :i+1])[0], 1)
		y_char.append(indices_char[ix[-1]])
	return ''.join(y_char)

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)

    start_index = random.randint(0, len(names) - maxlen - 1)

    for diversity in [0.6, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = names[start_index: start_index + maxlen]
        generated.join(sentence)
        print('----- Generating with seed: "')
        print(sentence)
        sys.stdout.write(generated)

        for iteration in range(140):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

model.save_weights('donald_weights.h5', overwrite=True)

"""def sample(a, temp=1.0):
	a = np.log / temp
	a = np.exp(a) / np.sum(np.exp(a))
	return np.argmax(np.random.multinomial(1, a, 1))

for iteration in range(1, 60):
	print()
	print('-' * 50)
	print('Iteration', iteration)
	model.fit(X, y, batch_size=128, nb_epoch=1)

	start_index = random.randint(0, len(names_file) - maxlen - 1)

	for diversity in [0.2, 0.5, 1, 1.2]:
		print()
		print('----diversity:', diversity)
		generated = ''
		name = names_file[start_index: start_index + maxlen]
		generated += name
		print('----Generating with seed: "' + name + "")
		sys.stdout.write(generated)

maxlen = 20
step = 3
names = []
next_chars = []
for i in range(0, len(str) - maxlen, step):
	names.append(str[i: i + maxlen])
	next_chars.append(str[i + maxlen])
print('nb sequences:', len(names)"""
