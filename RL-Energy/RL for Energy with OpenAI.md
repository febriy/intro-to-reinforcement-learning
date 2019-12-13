# RL for Energy with OpenAI

## Overview

### Details

- Name: GuestRoom-v0

- Category: HVAC system control

### Description

The testing environment will be a room of a specified size with an inflow of air-conditioned air and (undesired) exchange of air from outside of the room. The goal is to keep the room temperature within a certain treshold while minimising energy usage. 

## Observations

The environment's `step` function returns four values:

1. `observation`(object): an environment-specific object representing your observation of the environment. For example, temperature, energy used. 

2. `reward`(float): amount of reward achieved by the previous action. Reward can be made up of both temperature and energy: 1 for every minute the temperature is within the target temperature and 5 when a certain energy goal is achieved OR when total electricity consumption (or cost) in the hour is less than a certain treshold. 

3. `done`(boolean): whether it's time to `reset` the environment again. It is 'achieved' when either 
   
   1. Temperature is more than (set temperature) ±1°
   
   2. Energy usage per hour is more than (set value) ± (treshold range)
   
   3. Episode length is longer than a threshold e.g. 8 hours. 

4. `info`(dict): diagnostic information useful for debugging, not to be used by the agent. I'm still unsure what this might be for the machine. 

## Spaces

Every environment comes with an `action_space` and an `observation_space`. 

### Actions

Type: Discrete(6)

| Num | Action             |
| --- | ------------------ |
| 0   | Turn off AC        |
| 1   | Turn on AC         |
| 2   | Decrease Temp      |
| 3   | Increase Temp      |
| 4   | Decrease Fan Speed |
| 5   | Increase Fan Speed |

Note: We can start with 2 actions first

| Num | Action        |
| --- | ------------- |
| 0   | Decrease Temp |
| 1   | Increase Temp |

### Observation

Type: Box(2)

| Num              | Observation            | Min | Max  |
| ---------------- | ---------------------- | --- | ---- |
| 0                | Room Temperature       | 0   | 35°C |
| 1                | Total Energy Usage     | 0   | inf  |
| 2 (stretch goal) | Total Electricity Cost | 0   | inf  |

### Solved Requirements

Considered solved when the average reward is greater than or equal to (a set value) over (a set number) hours. 

## References

- OpenAI [documentation](http://gym.openai.com/docs/#environments)

- [Cartpole](https://github.com/openai/gym/wiki/CartPole-v0) environment documentation

- 
