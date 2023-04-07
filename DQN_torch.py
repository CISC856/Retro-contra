import torch
import torch.nn as nn



class DQN(nn.Module):

    def __init__(self, observations, actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Conv2d(observations[0], 32 , kernel_size= 8, stride=4)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.layer3 = nn.Linear(43200, actions)
        

    
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
      
        
        x = nn.functional.relu(self.layer2(x)) 
        
        x = x.view(x.size()[0], -1)
        x = self.layer3(x)
        
        return x


class DuelingDQN(nn.Module):

    def __init__(self, observations, actions):
        super(DuelingDQN, self).__init__()
        self.layer1 = nn.Conv2d(observations[0], 32 , kernel_size= 8, stride=4)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.layerval = nn.Linear(43200, 1)
        self.layeradv = nn.Linear(43200, actions)
        

    
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
      
        
        x = nn.functional.relu(self.layer2(x)) 
        
        x = x.view(x.size()[0], -1)
        V = self.layerval(x)
        A = self.layeradv(x)
        Q = V + (A - torch.mean(A, dim=1, keepdim=True))
        
        return Q

class NDQN(nn.Module):

    def __init__(self, observations, actions):
        super().__init__()
        self.layer1 = nn.Linear(observations, 24)
        self.layer2 = nn.Linear(24, 24)
        self.layer3 = nn.Linear(24, actions)
        

    
    def forward(self, x):
        x = nn.functional.relu(self.layer1(x))
      
        
        x = nn.functional.relu(self.layer2(x)) 
        
       
        x = self.layer3(x)
        
        return x

