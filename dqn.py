import random
import collections

import gym

SEED = 123
N_EPISODES = 430
TARGET_UPDATE = 1024
GAMMA = 0.999
BATCH_SIZE = 32
MEMORY_SIZE = 1024
EPSILON_DECAY = 1/10000
TEST_PLAYS = 10

env_name = 'CartPole-v1'
env = gym.make(env_name)

import torch

# Reproducibility
env.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
env.action_space.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(torch.nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.linear_1 = torch.nn.Linear(env.observation_space.shape[0], 64)
        self.activation_1 = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(64, env.action_space.n)

    def forward(self, x):
        l1 = self.linear_1(x)
        a1 = self.activation_1(l1)
        l2 = self.linear_2(a1)
        return l2

policy = Policy()
target = Policy()
target.load_state_dict(policy.state_dict())
optimizer = torch.optim.Adam(policy.parameters())
loss_function = torch.nn.MSELoss()

max_reward = 0
epsilon = 1.0
experience_replay = collections.deque(maxlen=MEMORY_SIZE)
for episode in range(N_EPISODES):
    observation = (torch.from_numpy(env.reset())
                   .float()
                   .to(device))
    done = False
    while not done:
        # Playing
        if random.random() < epsilon: # Exploration
            action = random.randint(0, env.action_space.n-1)
        else: # Greedy Policy
            with torch.no_grad():
                action = policy(observation).max(0)[1].item()
        next_observation, reward, done, info = env.step(action)
        next_observation = (torch.from_numpy(next_observation)
                            .float()
                            .to(device))
        experience_replay.append((observation, action, reward,
                                  next_observation if not done else None))
        observation = next_observation
        epsilon -= EPSILON_DECAY
        epsilon = max(0.1, epsilon)

        # Learning
        if len(experience_replay) > BATCH_SIZE:
            # Uniformly sample the experience replay
            # batch = [observations, actions, rewards, next_observations]
            batch = list(zip(*random.sample(experience_replay, BATCH_SIZE)))

            # Get values of chosen actions, state action values
            output = policy(torch.stack(batch[0]).to(device))
            actions = torch.Tensor(batch[1]).unsqueeze(1).long()
            state_action_values = output.gather(1, actions)

            # Calculate expected return with optimal state-action function
            output = torch.zeros(BATCH_SIZE, device=device)
            idxs = [idx for idx in range(BATCH_SIZE) if batch[3][idx] is not None]
            next_states = [batch[3][idx] for idx in range(BATCH_SIZE) if batch[3][idx] is not None]
            output[idxs] = target(torch.stack(next_states)).max(1)[0].detach()
            expected_state_action_values = torch.Tensor(batch[2]) + GAMMA * output

            # MSELoss
            loss = loss_function(expected_state_action_values.unsqueeze(1), state_action_values)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            # Limit gradient update to increase stability
            for p in policy.parameters():
                p.grad.data.clamp_(-1.0, 1.0)
            optimizer.step()

            step = optimizer.state[next(policy.parameters())]["step"]
            if not (step % TARGET_UPDATE):
                target.load_state_dict(policy.state_dict())

    # Test the current policy
    total_reward = 0
    for i in range(TEST_PLAYS):
        observation = env.reset()
        done = False
        while not done:
            observation = (torch.from_numpy(observation)
                           .float()
                           .to(device))
            with torch.no_grad():
                action = policy(observation).max(0)[1].item()
            observation, reward, done, info = env.step(action)
            total_reward += reward
    total_reward = total_reward / TEST_PLAYS
    if total_reward >= max_reward:
        max_reward = total_reward
        best_network_dict = policy.state_dict()
    print(f"Episode: {episode}, Total Reward: {total_reward} Max Reward: {max_reward}")

# Final Test
policy.load_state_dict(best_network_dict)
for i in range(TEST_PLAYS):
    observation = env.reset()
    done = False
    total_reward = 0
    while not done:
        observation = (torch.from_numpy(observation)
                       .float()
                       .to(device))
        with torch.no_grad():
            action = policy(observation).max(0)[1].item()
        observation, reward, done, info = env.step(action)
        env.render()
        total_reward += reward
    print(f"Game {i} Total Reward: {total_reward}")
