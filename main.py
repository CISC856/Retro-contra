import retro
import numpy as np
#import torch
#import torch.nn as nn
from Agent import DQNAgent
import os
import json
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()







color = np.array([224, 240, 74]).mean()

def preprocess_state(state):

    #crop and resize the image
    image = state[::2, ::2]

    #convert the image to greyscale
    image = image.mean(axis=2)

    #improve image contrast
    image[image==color] = 0

    #normalize the image
    image = (image - 128) / 128 - 1
    
    image = np.expand_dims(image.reshape(-1, 112, 120), axis=1)
    

    return image

def action_to_list(a):
    actions=[0,0,0,0,0,0,0,0,0]
    actions[a]=1
    return actions

def main():
    num_episodes = 501
    num_timesteps = 20000
    batch_size = 32
    state_size = (1, 112, 120)

    # load game

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MAINMDOEL_DIR=os.path.join(SCRIPT_DIR, "mainmodels_restrict")
    TARGETMDOEL_DIR=os.path.join(SCRIPT_DIR, "targetmodels_restrict")

    retro.data.Integrations.add_custom_path(
        os.path.join(SCRIPT_DIR, "Contra_Integrations")
        )
    print("Contra-Nes" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
    env = retro.make("Contra-Nes", inttype=retro.data.Integrations.ALL, scenario='scenario',state='Level1.state')
    print(env)
  

    action_size = env.action_space.n

    # define agent
    
    dqn = DQNAgent(state_space=state_size, action_space=action_size, gamma=0.99, lr=0.001)
   
    load=False
    done = False
    time_step = 0
    resultlist = []
    lossv=[]

    # load model
    if load:
        mainmodelfile='mainnetwork'+str(500)+'.pth'
        targetmodelfile='targetnetwork'+str(500)+'.pth'
        mainmodel_dir=os.path.join(MAINMDOEL_DIR, mainmodelfile)
        targetmodel_dir=os.path.join(TARGETMDOEL_DIR, targetmodelfile)
        dqn.loadmodel(mainmodel_dir,targetmodel_dir)

    #for each episode
    for i in range(0,num_episodes):
        
        #set return to 0
        Return = 0
        
        #preprocess the game screen
        state = preprocess_state(env.reset())

        average_loss = 0

        #for each step in the episode
        for t in range(num_timesteps):

            losslist = []
            #render the environment
            env.render(mode='rgb_array')
            #env.render()
            #update the time step
            time_step += 1
            
            #update the target network
            if time_step % dqn.update_rate == 0:
                dqn.update_target_network()
            
            #select the action
            action = dqn.epsilon_greedy(state)

            real_action=action_to_list(action)
            
            #perform the selected action
            next_state, reward, done, info = env.step(real_action)
            
            #preprocess the next state
            next_state = preprocess_state(next_state)
            
            #store the transition information for the experience replay
            dqn.store_transistion(state, action, reward, next_state, done)
            
            #update current state to next state
            state = next_state       
            
            #update the return
            Return += reward
            
            # train the network when reply buffer stores enough
            if len(dqn.replay_buffer) > batch_size:
                lossvalue = dqn.train(batch_size)
                writer.add_scalar("Loss/train", lossvalue, i)
                writer.flush()
                losslist.append(lossvalue.item())

            #record and output the some info
            if done or (t == num_timesteps -1):
                resultlist.append(Return)
                if losslist:
                    average_loss = np.mean(losslist)
                    lossv.append(average_loss)
                print('Episode: ',i, 'End in',t,'steps ','Return', Return, "LOSS",lossv[-1],"Progress:",info['scroll_center'])
                if i%10==0:
                    mainmodelfile='mainnetwork'+str(i)+'.pth'
                    targetmodelfile='targetnetwork'+str(i)+'.pth'
                    mainmodel_dir=os.path.join(MAINMDOEL_DIR, mainmodelfile)
                    targetmodel_dir=os.path.join(TARGETMDOEL_DIR, targetmodelfile)
                    dqn.savemodel(mainmodel_dir,targetmodel_dir)
                break
           
            
        writer.add_scalar("Loss/train", average_loss, i)
        writer.flush()    

    env.close()
    writer.close()
    
    json_object = json.dumps(resultlist, indent=4)
    with open("result.json", "w") as outfile:
        outfile.write(json_object)
    lossval=json.dumps(lossv,indent=4)
    with open("totalloss.json", "w") as outfile:
        outfile.write(lossval)

if __name__ == "__main__":
    main()
