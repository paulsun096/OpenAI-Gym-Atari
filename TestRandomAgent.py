import gym

# TODO: Load an environment
test_game  = "DemonAttack-ram-v0"
    #'CartPole-v1'
#"DemonAttack-ram-v0"
env = gym.make(test_game)

# TODO: Print observation and action spaces
print(env.action_space)
print(env.observation_space)

# # TODO Make a random aget
games_to_play = 100

for i in range(games_to_play):
    #reset the environment
    obs = env.reset()
    episode_rewards = 0
    done = False

    while not done:
        #render the environment
        env.render()

        #choose random action
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)
        episode_rewards += reward

    print(episode_rewards)

env.close()
