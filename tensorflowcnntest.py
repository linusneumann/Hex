import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import matplotlib.pyplot as plt
import re
import json


# transfrom data into 11x11x5 and in y_train and x_train
# 5 = 1 -black 2 white 3empty 4blackmove 5 whitemove


def parse_board_string(board_str):
    lines = board_str.strip().split('\n')  # jede Zeile einzeln
    board = []
    for line in lines:
        # extract numbers only 
        numbers = list(map(int, re.findall(r'-?\d+', line)))
        board.append(numbers)
    return np.array(board)

def parse_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    #extract pattern matrix,move,b or w
    pattern = r'\[([\s\S]*?)\],\s*(\d+),\s*([BW])'
    matches = re.findall(pattern, content)
    length = len(matches)
    parsed_data = []
    test_data = []
    for matrix_str, move, color in matches:
        board = parse_board_string("[" + matrix_str + "]")
        parsed_data.append((board, int(move), color))

    return parsed_data


def prepare_tensors13x13(parsed_data,test_ratio):
    x = []
    y = []
    x_test = []
    y_test = []
    
    for board, move, color in parsed_data:
        size = board.shape[0]
        padded_board = add_padding(board)
        only_black = add_padding((board ==1) * 1)
        only_white = add_padding((board == -1) * -1)

        # Farbebene: 1 für B, -1 für W
        color_plane = np.ones_like(padded_board) * (1 if color == 'B' else -1)

        next_player_plane = np.ones_like(padded_board) *(1 if color == 'B' else -1)

        # Input planes: shape (size, size, 3)
        planes = np.stack([padded_board, only_black, only_white, color_plane, next_player_plane], axis=-1)
       
        x.append(planes)
        #row,col = divmod(move,11)
        #move = row+1 *13 +col+1 
        y.append(move) # angenomen 0-120

    x = np.array(x,dtype=np.float32)
    y = np.array(y,dtype=np.int32)
        
    length = len(x)
    split = int(length * (1-test_ratio))
    x_train,y_train = x[:split],y[:split]
    x_test,y_test = x[split:],y[split:]
    # In Tensoren umwandeln
    x = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y = tf.convert_to_tensor(y_train, dtype=tf.int32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
    return x, y, x_test,y_test

def prepare_tensors(parsed_data,test_ratio):
    x = []
    y = []
    x_test = []
    y_test = []
    
    for board, move, color in parsed_data:
        size = board.shape[0]
       
       #optional 
        only_black = (board ==1) * 1
        only_white = (board == -1) * -1

        # Farbebene: 1 für B, -1 für W
        color_plane = np.ones_like(board) * (1 if color == 'B' else -1)

        next_player_plane = np.ones_like(board) *(1 if color == 'B' else -1)

        # Input planes
        planes = np.stack([board, only_black, only_white, color_plane, next_player_plane], axis=-1)
       
        x.append(planes)
        #row,col = divmod(move,11)
        #move = row+1 *13 +col+1 
        y.append(move) # angenomen 0-120

    x = np.array(x,dtype=np.float32)
    y = np.array(y,dtype=np.int32)
        
    length = len(x)
    split = int(length * (1-test_ratio))
    x_train,y_train = x[:split],y[:split]
    x_test,y_test = x[split:],y[split:]
    
    x = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y = tf.convert_to_tensor(y_train, dtype=tf.int32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)
    return x, y, x_test,y_test

def add_padding(board):
    padded_board = np.zeros((13,13),dtype=int)
    padded_board[1:-1,1:-1] = board

    #padding
    #oben und unten
    padded_board[0,1:-1] = 1
    padded_board[-1,1:-1] = 1
    #links und rechts
    padded_board[1:-1,0] = -1
    padded_board[1:-1,-1] = -1
    #ecken
    padded_board[0,0] = 1 
    padded_board[12,12] = 1 
    padded_board[0,12] = 1 
    padded_board[12,0] = 1 
    #print(padded_board)
    return padded_board

#--------------------------------------------------------------------------------------------------------
#
#       Create CNN
#       build after paper
#--------------------------------------------------------------------------------------------------------
def build_cnn(input_shape=(11, 11, 5), d=5, w=128, num_classes=121):
    inputs = tf.keras.Input(shape=input_shape)

    #  5x5 Conv layer with ReLU
    x = keras.layers.Conv2D(w, kernel_size=(5, 5), padding='same', use_bias=True, activation='relu')(inputs)

    # (d - 1) repetitions of 3x3 Conv + ReLU
    for i in range(d - 1):
        x = keras.layers.Conv2D(w, kernel_size=(3, 3), padding='same', use_bias=True, activation='relu')(x)

    # 1x1 Conv with position bias (assume just another Conv with bias)
    x = keras.layers.Conv2D(1, kernel_size=(1, 1), padding='same', use_bias=True)(x)

    # flatten and get ouput
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(num_classes)(x)
    outputs = keras.layers.Activation('softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    d = 5
    w = 128
    input_x = 11
    input_y = 11
    input_pad = 5
    cnn = build_cnn(input_shape=(input_x,input_y,input_pad),d=d,w=w)

    #for training 
    cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
    boards = parse_file("actions/v4-11x11-mohex-mohex-cg2010-vs-mohex-mohex-weak.txt")
    moves = [move for _, move, _ in boards]
    print("Verschiedene Züge:", sorted(set(moves)))
    random.shuffle(boards)
    x_train , y_train,x_test,y_test = prepare_tensors(boards,0.1)
    #y_train_one_hot = keras.utils.to_categorical(y_train,121)
    #y_test_one_hot = keras.utils.to_categorical(y_test,121)
    print(x_train.shape)
    print(y_train.shape)
    epochs = 30
    batchsize = 64
    validation_split = 0.2
    #print(np.array_equal(x_train[0], x_test[0]))
    callback = tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_accuracy',restore_best_weights=True)
    hist1 = cnn.fit(x_train,y_train, batch_size=batchsize,validation_split=validation_split, epochs=epochs,callbacks=[callback])
    cnn.save(f"4_{batchsize}-{epochs}-d{d}-cnntest{input_x}x{input_y}x{input_pad}.keras")
    #board = create_test_board()

    with open(f"{input_x}x{input_y}x{input_pad}_4_d{d}_w{w}_sparse_cce_dense_{batchsize}_acc1.json", "w") as f:
        json.dump(hist1.history, f)

    #cnn = keras.models.load_model("4_64-30-cnntest13x13x5.keras")
    """cnn.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy',keras.metrics.TopKCategoricalAccuracy(k=15)])"""
    test_loss, test_acc = cnn.evaluate(x_test, y_test)
    print(f"Testgenauigkeit: {test_acc:.4f}")
    print(f"Testverlust {test_loss:.4f}")
    #print(f"Top K Genauigkeit {test_topk:.4f}")
    
    


    