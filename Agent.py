import torch
import torch.nn as nn
import random
from DQN_torch import DQN, DuelingDQN, NDQN
import numpy as np
from collections import deque


class DQNAgent:

    def __init__(self, state_space, action_space, gamma, lr,
                 ):

        # load parameters
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.update_rate = 1000
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.l1 = nn.SmoothL1Loss().to(self.device) # Also known as Huber loss
        self.l2 = nn.MSELoss().to(self.device)
        self.epsilon = 0.1
        self.minepsilon=0.01
        self.lr = lr
        self.replay_buffer = deque(maxlen=2500)

        self.main_network = DuelingDQN(state_space, action_space).to(self.device)
        self.target_network = DuelingDQN(state_space, action_space).to(self.device)

        main_state_dict=self.main_network.state_dict()
        self.target_network.load_state_dict(main_state_dict)

    # experience replay pool
    def store_transistion(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

        

    # epsilon-greedy policy
    def epsilon_greedy(self, state):

        if random.uniform(0,1) < self.epsilon:
            return np.random.randint(self.action_space)
        
        state = torch.from_numpy(state).to(self.device).float()
        
        Q_values = self.main_network(state)
        self.epsilon*=0.999
        self.epsilon = max(self.epsilon, self.minepsilon)

        return torch.argmax(Q_values[0])
        
        


    
    #train the network
    def train(self, batch_size):
        
        #sample from the replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)

        state, action, reward, next_state, done = zip(*minibatch)
        
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.lr)

        self.optimizer.zero_grad()

        state = torch.FloatTensor(np.array(state)).squeeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).squeeze(1).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
     
        
        #compute the current Q value using the main network
        current_Q = self.main_network(state).gather(1, action.unsqueeze(1)).squeeze(1)


        #compute the target Q value using the target network
        target_Q = self.target_network(next_state).max(1)[0].detach()
        target_Q = reward + self.gamma * target_Q * (1 - done)

        loss = self.l2(current_Q, target_Q)
        torch.clamp(loss, min=-1, max=1)
        loss.backward()
        
        self.optimizer.step() 

        return loss
       
    def savemodel(self,path1,path2):
        torch.save(self.main_network.state_dict(),path1)
        torch.save(self.target_network.state_dict(),path2)
        
    def loadmodel(self,maindict,targetdict):
        self.main_network.load_state_dict(torch.load(maindict))
        self.target_network.load_state_dict(torch.load(targetdict))

    #update the target network weights by copying from the main network
    def update_target_network(self):
        #target_state_dict=self.target_network.state_dict()
        main_state_dict=self.main_network.state_dict()
        #for key in main_state_dict:
            #target_state_dict[key]=main_state_dict[key]
        self.target_network.load_state_dict(main_state_dict)
        #self.target_network.weight = nn.Parameter(self.main_network.parameters())