# Retro-contra
```
Depedencies: python 3.6-3.8
Follow the https://github.com/openai/retro instructions
To find out how to integrate the games: go to https://retro.readthedocs.io/en/latest/
```
To replicate our results:
1. Integrate the game according to this link:
 https://retro.readthedocs.io/en/latest/integration.html#using-a-custom-integration-from-python
2. Ensure the repository remains the same structure as it lay out in this repository
3. change the code in main.py 

The game rom is provided in the Contra_Integrations folder as rom.nes
```
Transfer learning:
The script sets that for 10 timesteps it saves the model in the directory(target network and main network separately).
For line 69, change load=True
For line 77 & 78, change the variable to the name you give to the models
In this case, if you want to invoke the model we give:
mainmodelfile='mainnetwork'+str(500)+'.pth'
targetmodelfile='targetnetwork'+str(500)+'.pth'
```
The model we give contains the weights for the Dueling DQN.
For Double DQN, it cannot work.</br>

If you want to train for Double DQN, please try to modify line 27 and 28 to </br>
self.main_network = DQN(state_space, action_space).to(self.device)</br>
self.target_network = DQN(state_space, action_space).to(self.device)

```
To do the training:
Run python main.py
```

