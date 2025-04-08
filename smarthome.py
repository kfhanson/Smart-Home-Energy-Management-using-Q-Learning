import pandas as pd
import numpy as np
import random

df = pd.read_csv('Homework_3_Dataset_Smart_home 1.csv')
df['Next Time'] = df['Time'].shift(-1)
df.loc[df.index[-1], 'Next Time'] = df.loc[df.index[0], 'Time']
df['Next Time'] = df['Next Time'].astype(int)
df['Next Electricity Price'] = df['Electricity Price'].shift(-1)
df.loc[df.index[-1], 'Next Electricity Price'] = df.loc[df.index[0], 'Electricity Price']
df['Next User Activity'] = df['User Activity'].shift(-1)
df.loc[df.index[-1], 'Next User Activity'] = df.loc[df.index[0], 'User Activity']
df['Next User Activity'] = df['Next User Activity'].astype(int)

time = 24
user_activity = 2
appliance_state = 2
action = 2

epsilon = 1.0 # exploration rate for ɛ-Greedy
epsilon_min = 0.01
decay = 0.999

q_table = np.zeros((time, user_activity, appliance_state, action))
total_reward = []

for epoch in range(1000):
    current_appliance_state = random.choice([0, 1])
    epoch_reward = 0
    for index, row in df.iterrows():
        current_time = int(row['Time'])
        current_user_activity = int(row['User Activity'])
        state = (current_appliance_state, current_user_activity, current_appliance_state)
        state_idx = state
        if random.uniform(0, 1) < epsilon:
            action = random.choice([0, 1])
        else:
            action = np.argmax(q_table[state_idx])
        next_time = int(row['Next Time'])
        next_user_activity = int(row['Next User Activity'])
        next_electricity_price = row['Next Electricity Price']
        next_appliance_state = action
        next_state = (next_time, next_user_activity, next_appliance_state)
        next_state_idx = next_state
        
        reward = 0
        if next_appliance_state == 1:
            reward = -next_electricity_price
        elif next_appliance_state == 0 and next_user_activity == 1:
            reward = -0.5 #comfort penalty
        epoch_reward += reward
        old_value = q_table[state_idx + (action,)]
        next_max_q = np.max(q_table[next_state_idx])
        target = reward + 0.95 * next_max_q # 0.95 as gamma or discount factor
        new_value = old_value + 0.1 * (target - old_value) # 0.1 as alpha or learning rate
        q_table[state_idx + (action,)] = new_value
        current_appliance_state = next_appliance_state
    if epsilon > epsilon_min:
        epsilon *= decay
    total_reward.append(epoch_reward)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/1000, Reward: {epoch_reward:.2f}, Eps: {epsilon:.3f}")

for user_act in [0, 1]:
    for app_state in [0, 1]:
        state_idx = (12, user_act, app_state)
        action = np.argmax(q_table[state_idx])
        print(f" State (T=12, User={user_act}, Appliance={app_state}): Recommended action = {action} (Q-vals: {q_table[state_idx].round(3)})")

total_cost = 0
comfort_violations = 0
agent_schedule = []
current_appliance_state = 0
cumulative_reward = 0

for index, row in df.iterrows():
    current_time = int(row['Time'])
    current_user_activity = int(row['User Activity'])
    current_price = row['Electricity Price']
    next_user_activity = int(row['Next User Activity'])
    next_price = row['Next Electricity Price']
    state_idx = (current_time, current_user_activity, current_appliance_state)
    action = np.argmax(q_table[state_idx])
    agent_schedule.append(action)
    next_appliance_state_actual = action
    if next_appliance_state_actual == 1:
        cost_incurred = next_price
        total_cost += cost_incurred
        reward = -cost_incurred
    else:
        reward = 0
        if next_user_activity == 1:
            comfort_violations += 1
            reward = -0.5
    
    cumulative_reward += reward
    current_appliance_state = next_appliance_state_actual

# Evaluation
print(f"  Total Electricity Cost: ${total_cost:.2f}")
print(f"  Number of Comfort Violations (Appliance OFF when User Active): {comfort_violations}")
print(f"  Total Reward (during evaluation): {cumulative_reward:.2f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(total_reward)
plt.title('Total Reward per Epoch during Training')
plt.xlabel('Epoch')
plt.ylabel('Total Reward')
plt.grid(True)
if len(total_reward) >= 50:
    moving_avg = pd.Series(total_reward).rolling(50).mean()
    plt.plot(moving_avg, label='50-Epoch Moving Avg', color='red', linewidth=2)
    plt.legend()
plt.show()