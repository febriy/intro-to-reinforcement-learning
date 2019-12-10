# Reinforcement Learning

### By [Sentdex](https://pythonprogramming.net/q-learning-algorithm-reinforcement-learning-python-tutorial/)



## Part 1 & 2

### Q-Learning

- Reinforcement learning agent seek maximum reward

- Q is the measure of cumulative value of an action

- An agent will choose an action that it predicts will give the maximum Q value

- The mechanics of choosing the Q-value is facilitated through a Q-value matrix, where the rows and columns correspond to a particular state and action

### Walkthrough

First, we will initialise a pre-set environment

```
env = gym.make("MountainCar-v0")
```

We will set several variables:

- Learning rate is used to update the Q-value matrix

- Discount affects how much future reward is considered by the model

- Episodes is the number of distinct sessions that will be allocated for the agent to learn and the model be trained

```
DISCOUNT = 0.95
EPISODES = 25000
```



The agent created can be made to prefer exploitation or exploration. Exploitative agent prefers paths or actions that it has alredy learnt to be beneficial, while explorative agent might forgo immediate reward in pursuit of another action that might seemingly not help in the short term but will enhance future reward. Epsilon is a measure of the agent's preference for exploration vs exploitation

```
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
```

#### Creating Q-table

The Q-table shows the Q value for every state - action combination. How many states do we want to use for our model? We can start by querying the possible combination of states:

```
print(env.observation_space.high)
print(env.observation_space.low)
```

We want to limit the state combinations to make the problem more computationally managable. 

Now, we will create the Q-table structure. We set an abritrary number 20 x 20. This means that we are restricting the action-value combination to be 20 x 20 (how does this make sense if there is only 3 actions? What does the size mean?)

We also find out the discrete window size, the gap between the window value.

```
DISCRETE_OS_SIZE = [20, 20]discrete_os_win_size = (    env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
```

Now we initialise our Q-table with random number. The code below will create a table of 20x20x3 shape. Note that the 3 corresponds to the 3 actions that the agent can take. The choice of -2 to 0 is because the reward given to the car is either -1 or 0. Negative values seems to make sense in this case, but nothing is stopping you from trying out other values. 

QN: I'm not sure about the numpy table size setting. Consult cheatsheet.

```
q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])
)
```

Now we can proceed with training our model, on in other words, update our Q-table. We loop through the training sequence,

```
for episode in range(EPISODES):
```

Initialise discrete_state to correspond to the banding made earlier.

```
discrete_state = get_discrete_state(env.reset())
```

#### Taking actions

We can set up a function to encourage or discourage exploration by the agent. Note how agent will either choose known knowledge (choosing action with known highest reward) or decide a random action. 

```
if np.random.random() > epsilon:
    # Get action from Q table
    action = np.argmax(q_table[discrete_state])
else:
    # Get random action
    action = np.random.randint(0, env.action_space.n)
```

#### Actions beget new state and reward

Every action taken by the agent will produce new state and reward, which we will then convert to conform to the new discrete states

```
new_state, reward, done, _ = env.step(action)
new_discrete_state = get_discrete_state(new_state)
```

If we print out the reward, and state, we will learn that the reward given in any state is -1 and 0 only when the car reaches the flag. We also see that the state can be represented by the location and the velocity of the car. 

QN: I would have thought that location should be in x and y direction. However this is not the case here. Find out more!

```
-1.0 [-0.26919024 -0.00052001]
-1.0 [-0.27043839 -0.00124815]
-1.0 [-0.2724079  -0.00196951]
-1.0 [-0.27508804 -0.00268013]
```

#### Continuous loop will update Q-table

It is now time to train the model. 

```
if not done:
```

Model will look at the maximum Q value in the next step (a new state)

```
max_future_q = np.max(q_table[new_discrete_state])
```

It will also look at the Q value for current state and performed action.

```
current_q = q_table[discrete_state + (action,)]
```

This is our equation to determine new Q

```
new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                reward + DISCOUNT * max_future_q
```

Now we update the Q table with the new Q 

```
q_table[discrete_state + (action,)] = new_q
```



## Full Code

```
# objective is to get the cart to the flag.
# for now, let's just move randomly:
import gym
import numpy as np

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 3000

DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table


for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done:

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)


        new_state, reward, done, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)

        if episode % SHOW_EVERY == 0:
            env.render()
        #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q


        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            #q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

env.close()
```


