from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)),allow_pickle=True)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()                                                                 # Initialize the Sequential model
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(50,63)))   # Add the first LSTM layer with 64 units
model.add(LSTM(128, return_sequences=True, activation='relu'))                       # Add the second LSTM layer with 128 units
model.add(LSTM(64, return_sequences=False, activation='relu'))                       # Add the third LSTM layer with 64 units
model.add(Dense(64, activation='relu'))                                              # Add a Dense layer with 64 units
model.add(Dense(32, activation='relu'))                                              # Add another Dense layer with 64 units
model.add(Dense(actions.shape[0], activation='softmax'))                             # Add the output Dense layer with the no. of units equal to no. of actions                                                                                   
res = [.7, 0.2, 0.1]
# Compile the model with Adam optimizer, categorical crossentropy loss, and track categorical accuracy
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# Fit the model with training data, specify number of epochs, and include a callback for tensorboard
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
model.summary()     # Print the model summary to display the model architecture

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')