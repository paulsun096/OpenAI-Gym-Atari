# OpenAI-Gym-Atari
Implementation of OpenAI's Gym in Python.
<div>
  <h2>TestRandomAgent.py</h2>
  <p>
    For Atari 2600 games, load gym environments that end with "-ram". <br> <br>
    Set the environment variable to names that end in "-ram" as such: test_game = "MsPacman-ram-v0" <br><br>
                                               instead of:            test_game = "MsPacman-v0" <br><br>
    since the Atari gym envrionment's take the action_space and observation_spaces from the "-ram" version of Atari games. <br><br>
    It's just how OpenAI works, check out the link <a href="https://gym.openai.com/envs/#atari" placeholder="here">here</a>
    to see the OpenAI Gym documentation and other Atari games. 
  </p>
</div>
