import sys

from snake_env import SnakeEnv
import numpy as np
from DQNetwork import DQNetwork
from Results import Results, test

import pickle

if len(sys.argv)>4:
    nrow, ncol = int(sys.argv[4]), int(sys.argv[4])
else:
    nrow, ncol = 5,5
model = DQNetwork(4, (1,nrow, ncol))
env= SnakeEnv(nrow, ncol, colors = 'gray')



if len(sys.argv)>5:
    n_train = int(sys.argv[5])
else:
    n_train = 1000
n_batch = 5000


episode_count = 500
imax = 100
res = Results()

loadModel = False
results_filename = 'results.pkl'
model_filename = 'model.h5'

print(sys.argv)
if len(sys.argv)>1:
    if sys.argv[1]=='load':
        loadModel = True
    results_filename = sys.argv[2]
    model_filename = sys.argv[3]
    
if loadModel:
    model.load(model_filename)
    with open(results_filename, 'rb') as input:
            res = pickle.load(input)
    epsilon = res.epsilon
    epsilons = np.maximum(np.arange(epsilon,epsilon -0.9, - 0.9/n_train), 0.1)
else:
    epsilons = np.arange(1,0.1, - 0.9/n_train)

# size of memory
N = 100000
for i_train in range(n_train):
    epsilon = epsilons[i_train]
    res.epsilon = epsilon
    print("Training: round {}, epsilon = {}".format(i_train, round(epsilon,2)))
    lengths_i_train = []
    scores_i_train = []
    for i_episode in range(episode_count):
        i=0
        done = False
        grid = env.reset()
        grid= grid.reshape((1,1, env.nrow, env.ncol))
        while i <imax:
            i+=1
            source = grid.copy()
            if epsilon >= np.random.rand():
                action = np.random.randint(4)
            else:
                action = np.argmax(model.predict(source))
            grid, reward, done = env.step(action)
            grid= grid.reshape((1,1, env.nrow, env.ncol))
            observation = {'source':source, 'action':action, \
                           'dest':grid, 'reward':reward,'final':done}
            res.memory.append(observation)
            if done:
                break
            
        lengths_i_train.append(i)
        scores_i_train.append(env.score)
        
    res.lengths.append(lengths_i_train)
    res.scores.append(scores_i_train)

    res.memory = res.memory[-N:]
    #print("Mean length of episode:", np.mean(lengths[i_train]))
    #print("Mean score:", np.mean(scores[i_train]))
    if i_train %10 ==0:
        (l, s) = test(env,model)
        print("Test: mean length {}, mean score {}".format(l,s))
        res.lengths_expl.append(l)
        res.scores_expl.append(s)
        
        # save results
        with open(results_filename, 'wb') as output:
            pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)
        
        # save model
        model.save(model_filename)
        
    if len(res.memory)>=n_batch:
        i_batch = np.random.choice(np.arange(0, len(res.memory)), n_batch, replace = False)
        batch = [res.memory[i] for i in i_batch]
    else:
        batch = res.memory
        
    model.train(batch)       
