#MODIFIED = not included in gameplan code

import tensorflow as tf
import gym
import numpy as np
import os

#MODIFIED
from Agent import Agent

# TODO Create the discounted and normalized rewards function

discount_rate = 0.95

def discount_normalize_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    total_rewards = 0

    for i in reversed(range(len(rewards))):
        total_rewards = total_rewards * discount_rate + rewards[i]
        discounted_rewards[i] = total_rewards

    discounted_rewards -= np.mean(discounted_rewards)
    # cannot divide by zero

    if np.std(discounted_rewards) != 0:
        discounted_rewards /= np.std(discounted_rewards)

    return discounted_rewards

game_name = "CartPole-v1"

env = gym.make(game_name)

print('state size')
print(env.observation_space)
print('action space')
print(env.action_space)
print("")
print("--------------------------")
print("")

tf.reset_default_graph()

# Modify these to match shape of actions and states in your environment

if game_name == "CartPole-v0":
    num_actions = 2
    state_size = 4

'''
if game_name == "CrazyClimber-ram-v0":
    num_actions = 9
    state_size = 128

'''

path = "./"+game_name+"/"

training_episodes = 2500
max_steps_per_episode = 10000
episode_batch_size = 5

HID_LAYERS = 2

init = tf.global_variables_initializer()

num_actions = 2
state_size = 4

agent = Agent(num_actions, state_size, HID_LAYERS)

saver = tf.train.Saver(max_to_keep=2)

if not os.path.exists(path):
    os.makedirs(path)

def train():

    with tf.Session() as sess:

        #MODIFY: Initialize variables before running train loop
        tf.initialize_all_variables().run()

        sess.run(init)

        total_episode_rewards = []
        gradient_buffer = sess.run(tf.trainable_variables())

        #MODIFY

        #Episodes per step, or episode checkpoint: each episode's history is stored step by step (in this case, every 100 steps)
        eps = 100

        for index, gradient in enumerate(gradient_buffer):
            gradient_buffer[index] = gradient * 0

        for episode in range(training_episodes):

            print("--------------------------------")
            print("Running training episode: ")
            print(episode)
            print("")


            print("Resetting environment state: ")
            state = env.reset()
            print("")

            episode_history = []
            episode_rewards = 0

            for step in range(max_steps_per_episode):

                print("Step: ")
                print(step)
                print("")

                print("For episode: ")
                print(episode)
                print("")

                if episode % eps == 0:
                    env.render()

                #get weights for each action
                print("Getting weights for each action...")
                print("")

                action_probabilities = sess.run(agent.outputs, feed_dict={agent.input_layer: [state]}) # tried [None, state]

                print("Action Probabilities: ")
                print(action_probabilities)
                print("")

                print("action_probabilities[0]: ")
                print(action_probabilities[0])
                print("")

                #num_actions: 6 for DemonAttack
                action_choice = np.random.choice(range(num_actions), p=action_probabilities[0])

                print("Action Choice: ")
                print(action_choice)

                #step returns: observation (object), reward (float), done (boolean), info (dict)
                state_next, reward, done, _ = env.step(action_choice)

                print("")
                print("done: ")
                print(done)

                print("")
                print("State: ")
                print(state)

                print("")
                print("state_next: ")
                print(state_next)

                episode_history.append([state, action_choice, reward, state_next])

                '''
                print("")
                print("'episode_history' ")
                print(episode_history)
                '''

                state = state_next

                episode_rewards += reward
                print("")
                print("episode rewards: ")
                print(episode_rewards)

                if done:
                    state = env.reset()

                if done or step + 1 == max_steps_per_episode:

                    total_episode_rewards.append(episode_rewards)
                    episode_history = np.array(episode_history)

                    '''
                    print("")
                    print("'np.array(episode_history)' ")
                    print(episode_history)
                    '''

                    print("")
                    print("'episode_history[:,2]' ")
                    print(episode_history[:,2])

                    episode_history[:,2] = discount_normalize_rewards(episode_history[:,2])

                    print("")
                    print("'episode_history[:,2] after passing into discount normalize rewards: ' ")
                    print(episode_history[:, 2])

                    #Episode gradients, or weights.
                    ep_gradients = sess.run(agent.gradients, feed_dict={agent.input_layer: np.vstack(episode_history[:, 0]),
                                                                        agent.actions: episode_history[:,1],
                                                                        agent.rewards: episode_history[:,2]})
                    #Add the episode gradients to the gradient buffer
                    for index, gradient in enumerate(ep_gradients):
                        gradient_buffer[index] += gradient

                    break

            #After the step for loop, check whether agent is ready for update

            if episode % episode_batch_size == 0:

                feed_dict_gradients = dict(zip(agent.gradients_to_apply, gradient_buffer))

                sess.run(agent.update_gradients, feed_dict=feed_dict_gradients)

                for index, gradient in enumerate(gradient_buffer):
                    gradient_buffer[index] = gradient * 0

                if episode % eps == 0:
                    #pg-checkpoint, (policy gradient checkpoint)
                    saver.save(sess, path + "pg-checkpoint", episode)

                    # MODIFY
                    #eps = episodes per step
                    print('Average reward /' + str(eps) + ' episodes per step: ' + str(np.mean(total_episode_rewards[-100:])))

def prompt():

    ask = input('Do you want to train? (y/n): ')
    if ask=='y':
        train()
    elif ask=='n':
        print('Program not training.')
    else:
        prompt()

prompt()