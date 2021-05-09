# neural network:
# take game states with PID as input and return a probability distribution over all possible moves as output
# output layer of network: fixed number of neurons: one for each possible move in a game
# use softmax as last activation function, but rescale over the right number of possible moves

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import math
from keras.models import model_from_json
import matplotlib.pyplot as plt


class Actor:
    def __init__(self, config, mapping, epsilon_decay=None, params=None, weights=None):
        if params is None:
            self.epsilon = config["epsilon"]
            # Need to be precomputed depending on num_actual_games to be played
            self.epsilon_decay = epsilon_decay
            self.learning_rate = config["learning_rate"]
            self.dims = self.create_dims(
                internal_dims=config["internal_dims"], mapping=mapping)
            self.batch_size = config["batch_size"]
            self.replay_buffer_max_len = config["replay_buffer_max_len"]
            self.opt = config["opt"]
            self.activation = config["activation"]
            self.model = self.gennet(self.dims, self.learning_rate)
            self.loss_history = []
            self.replay_buffer = []
        else:
            # Load saved model from incoming files
            self.load(params, weights)
        self.mapping = mapping

    @staticmethod
    def create_dims(internal_dims, mapping):
        return [int(math.sqrt(len(mapping))) ** 2 + 1] + internal_dims + [len(mapping)]

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def visualize_loss(self):
        x = [i for i in range(1, len(self.loss_history) + 1)]

        plt.title("Actor loss")
        plt.xlabel("Episodes")
        plt.ylabel("KLDivergence loss")
        plt.plot(x, self.loss_history)
        plt.show()

    @staticmethod
    def deepnet_cross_entropy(targets, outs):
        def safelog(tensor, base=0.0001):
            return tf.math.log(tf.math.maximum(tensor, base))

        return tf.reduce_mean(tf.reduce_sum(-1 * targets * safelog(outs), axis=[1]))

    def gennet(self, dims, learning_rate, loss="KLDivergence()", last_activation="softmax"):
        model = keras.models.Sequential()
        opt = eval('keras.optimizers.' + self.opt)
        loss = loss if type(loss) != str else eval('tf.keras.losses.' + loss)
        model.add(keras.layers.Dense(input_shape=(dims[0],),  # Determines shape after first input of a board state
                                     units=dims[0], activation=self.activation))
        for layer in range(1, len(dims) - 1):
            model.add(keras.layers.Dense(
                units=dims[layer], activation=self.activation))
        model.add(keras.layers.Dense(
            units=dims[-1], activation=last_activation))
        model.compile(optimizer=opt(lr=learning_rate),
                      loss=loss, metrics=['accuracy'])
        return model

    def convert_list_to_tensor(self, list_to_tensor):
        flat_array = np.array(list_to_tensor)
        return tf.reshape(flat_array, [1, len(flat_array)])

    def train(self, epochs=10):
        replay_buffer = self.get_batch()
        x = []
        y = []
        for state, action_dist in replay_buffer:
            p_d = np.zeros(self.dims[-1])
            # x.append(self.convert_list_to_tensor(state))
            x.append(np.array(state))
            for action in action_dist.keys():
                action_index = self.mapping[action]
                np.put(p_d, action_index, action_dist[action])
            y.append(np.array(p_d))
        x = tf.stack(x)
        y = tf.stack(y)
        self.model.fit(x=x, y=y, epochs=epochs, batch_size=16)

        # print(f"Loss: {self.model.evaluate(x,y)}")
        self.loss_history.append(self.model.evaluate(x, y)[0])

    def specialized_train(self, x, y):  # Did not end up using this
        with tf.GradientTape(persistent=True) as tape:
            prediction = self.model(tf.convert_to_tensor(x))
            target = y
            new_prediction = self.modify_prediction(prediction, target)
            target = tf.convert_to_tensor(target)
            target = tf.stop_gradient(target)

            loss = self.model.loss(y_true=target, y_pred=new_prediction)
            loss = tf.math.divide(loss, len(x))  # Avg loss per example
            # print("Specific loss:", loss)
            # self.loss_history.append(loss)

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables))
        return loss

    def modify_prediction(self, preds, targs):
        """
        Incoming preds is Eagertensor object
        """
        #  pred:  [0.2, 0.3, 0.4, 0.1]
        #  targ:  [0. , 0. , 0.2, 0.8]
        #  return [0. , 0. , 0.8, 0.2]

        masked = tf.where(targs == 0, 0, preds)
        norm = masked / tf.math.reduce_sum(masked)
        return norm

    def get_action(self, state, legal_actions, argmax=True, epsilon_greedy=False):
        if epsilon_greedy and random.randint(0, 1000)/1000 < self.epsilon:
            return random.choice(legal_actions)
        state_tensor = self.convert_list_to_tensor(state)
        prediction = self.model(state_tensor).numpy()[0]
        if argmax:
            return max(legal_actions, key=lambda action: prediction[self.mapping.get(action)])
        weights = np.array([prediction[self.mapping[action]]
                            for action in legal_actions])
        weights = weights / np.sum(weights)
        return random.choices(list(legal_actions), weights=weights, k=1)[0]

    @staticmethod
    def save(model, filename):
        model_jason = model.to_json()
        with open(filename + ".json", "w") as json_file:
            json_file.write(model_jason)
        model.save_weights(filename + ".h5")
        print("Model saved to disk")

    def get_batch(self):
        if self.batch_size < 1:  # Then interpret as percentage of rbuf
            batch_size = int(len(self.replay_buffer)*self.batch_size)
        else:
            batch_size = self.batch_size
        if len(self.replay_buffer) <= batch_size:
            return self.replay_buffer
        else:
            weights = np.linspace(start=0, stop=1, num=len(self.replay_buffer))
            weights = weights / np.sum(weights)
            index_list = [i for i in range(len(self.replay_buffer))]
            chosen_idxs = np.random.choice(
                index_list, batch_size, replace=False, p=weights)
            return np.array(self.replay_buffer, dtype=object)[chosen_idxs]

    def add_to_replay(self, case):
        self.replay_buffer.append(case)
        if len(self.replay_buffer) > self.replay_buffer_max_len:
            self.replay_buffer.pop(0)  # Maintain replay buffer at max length

    def print_accuracy(self):
        corrects = 0
        for state, action_dist_targets in self.replay_buffer:
            performed_action = max(
                action_dist_targets.keys(), key=lambda a: action_dist_targets[a])
            legals = action_dist_targets.keys()
            predicted_action = self.get_action(state, legals)
            if predicted_action == performed_action:
                corrects += 1
        print(
            f"Accuracy on this replay buffer was {corrects / len(self.replay_buffer)}")

    def load(self, parameters, weights):
        json_file = open(parameters, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weights)
        self.model = loaded_model
