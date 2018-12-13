# TODO Build the policy gradient neural network
import tensorflow as tf

class Agent:

    # Neural net starts here
    # MODIFY: include third param - hidden layers. Needed when called from an outside class to specify number of hidden layers

    def __init__(self, num_actions, state_size, num_hidden_layers):

        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, state_size])

        # MODIFY: hidden layers generated in for loop function below
        self.final_hidden_layer = Agent.make_hidden_layers(self.input_layer, num_hidden_layers)

        # Output of neural net
        out = tf.layers.dense(self.final_hidden_layer, num_actions, activation=None)

        self.outputs = tf.nn.softmax(out)

        '''
        print("Softmax Output Values for Agent: ")
        print(self.outputs)
        print("")
        '''

        print("Attempting to train Neural Network using " + str(num_hidden_layers) + " hidden layers..")
        print("")

        # what is the axis referring to?
        self.choice = tf.argmax(self.outputs, axis=1)

        # TRAINING PROCEDURE

        self.rewards = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32)

        one_hot_actions = tf.one_hot(self.actions, num_actions)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions)

        self.loss = tf.reduce_mean(cross_entropy * self.rewards)

        self.gradients = tf.gradients(self.loss, tf.trainable_variables())

        # placeholder list for gradients (functions)
        self.gradients_to_apply = []

        # what determines the number of trainable variables?

        for index, variable in enumerate(tf.trainable_variables()):
            gradient_placeholder = tf.placeholder(tf.float32)
            self.gradients_to_apply.append(gradient_placeholder)

        # operation updates gradients with gradient placeholders
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        self.update_gradients = optimizer.apply_gradients(zip(self.gradients_to_apply, tf.trainable_variables()))

    # This function creates a specified number of hidden layers.

    def make_hidden_layers(input_layer, num_layers):

        # The initial element in list is always the input layer
        layers = [input_layer]
        hidden_layer = None

        # For loop populates an array of hidden layers, starting with
        for i in range(num_layers):
            hidden_layer = tf.layers.dense(layers[i], 8, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            layers.append(hidden_layer)

        return hidden_layer

