from pendulum_problem import *
import torch
from torch import nn

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.sm = nn.Softmax()
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = self.sm(self.fc3(x))
        return action
    
class CriticNewtwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class PolicyGradientAgent():
    def __init__(self, state_dim, action_dim, lamda_w, lamda_theta, alpha_w, alpha_theta, time_step):
        self.lamda_w = lamda_w
        self.lamda_theta = lamda_theta
        self.alpha_w = alpha_w
        self.alpha_theta = alpha_theta
        self.time_step = time_step

        self.actor = ActorNetwork(state_dim, action_dim)
        print(self.actor)
        self.critic = CriticNewtwork(state_dim, action_dim)
        
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lamda_theta)
        # self.critic_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lamda_w)
        
        self.cart = PendulumCart()

    def take_action(self, state, action_prob):
        idx_action = torch.multinomial(action_prob, 1).item()
        return self.cart.next_state(state, self.time_step, idx_action), idx_action
    
    def observe_reward_terminated(self, state):
        if torch.abs(state[2]) > torch.pi / 2:
            return -100, True
        else:
            return 1, False
        # return -state[2]**2 - 0*state[0]**2
    
    def init_epsiode(self, initial_state = [0, 0, 0, 0]):
        state = torch.tensor(initial_state, dtype=torch.float32, requires_grad=True)
        self.state_value_eligibility = {name:torch.zeros_like(value) for name, value in self.critic.named_parameters()}
        self.actor_eligibility = {name:torch.zeros_like(value) for name, value in self.actor.named_parameters()}
        self.I = 1
        return state
    
    def update(self, state):
        action_prob = self.actor(state) # Compute the probabilities of the possible action according to policy
        # print(action_prob)
        
        value = self.critic(state)
    
        next_state, action = self.take_action(state.detach().numpy(), action_prob) # Take the action and store the next state
        next_state = torch.tensor(next_state, dtype=torch.float32, requires_grad=True)
        R, terminated = self.observe_reward(next_state) # Observe reward of the next state        

        # Compute eligibility traces according to previous state S
        value.backward()
        for name, param in self.critic.named_parameters():
            self.state_value_eligibility[name] = self.state_value_eligibility[name] * self.lamda_w + param.grad
        
        log_action_prob = torch.log(action_prob[action])
        log_action_prob.backward()
        for name, param in self.actor.named_parameters():
            self.actor_eligibility[name] = self.actor_eligibility[name] * self.lamda_theta + param.grad

        if not terminated:
            next_value = self.critic(next_state)
        else:
            next_value = 0

        delta = R - self.R_bar + next_value - value
        self.R_bar += self.alpha_R * delta

        # Modify the weights of the networks
        for name, param in self.critic.named_parameters():
            # print(delta)
            param.data.add(self.alpha_w * delta * self.state_value_eligibility[name])

        for name, param in self.actor.named_parameters():
            # print(self.alpha_theta * delta * self.actor_eligibility[name])
            param.data.add(self.alpha_theta * delta * self.actor_eligibility[name])
        
        return next_state, action_prob, terminated

num_episodes = 1000

if __name__ == '__main__':
    print("RUNNING")
    agent = PolicyGradientAgent(4, 2, lamda_w = 0.01, lamda_theta = 0.01, alpha_w = 0.001, alpha_theta = 0.001)
    state = initial_state
    for idx_episode in range(num_episodes):
        state = agent.init_epsiode(initial_state)
        step = 0
        terminated = False
        while not terminated:
            step += 1
            state, action_probabilities, terminated = agent.update(state)
            print(f"Step: {step}, Episode: {idx_episode}, State: {state.detach().numpy()}, Action probabilities: {action_probabilities.detach().numpy()}")
            