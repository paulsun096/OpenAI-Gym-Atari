import Training
import tensorflow as tf
import gym

testing_episodes = 100

env = gym.make(Training.game_name)

with tf.Session() as sess:
    print("Path for checkpoints: ")
    print(Training.path)

    checkpoint = tf.train.get_checkpoint_state(Training.path)
    Training.saver.restore(sess,checkpoint.model_checkpoint_path)

    for episode in range(testing_episodes):

        print("Path for checkpoints: ")
        print(Training.path)

        state = env.reset()

        episode_rewards = 0

        for step in range(Training.max_steps_per_episode):

            env.render()
            print('step: ' + str(step))
            print('----------------------------------------------')
            #Pass state into model, run choice operations
            print('Printing values for Training.agent.choice: ')
            print(Training.agent.choice)
            print('')

            #problem may be comming from sess.run() parameters
            action_argmax = sess.run(Training.agent.choice, feed_dict={Training.agent.input_layer: [state]})

            action_choice = action_argmax[0]

            print("action_choice: ")
            print(action_choice)
            print('')

            state_next, reward, done, _ = env.step(action_choice)
            state = state_next

            print("state: ")
            print(state)
            print('')

            print("reward: ")
            print(reward)
            print('')

            print("done: ")
            print(done)
            print('')

            episode_rewards += reward

            #MODIFY
            print("episode_rewards at step " + str(step) + ": " )
            print(episode_rewards)
            print('')
            print('----------------------------------------------')

            if done or step + 1 == Training.max_steps_per_episode:
                print("Rewards for episode " + str(episode) + ": " + str(episode_rewards))
                break
