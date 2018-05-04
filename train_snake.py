import sys

from snake_env import SnakeEnv
import numpy as np
from DQNetwork import DQNetwork
from Results import Results, test
import Options
import pickle


opt = Options().parse()
nrow, ncol = opt.gridsize, opt.gridsize
model = DQNetwork(4, (1,nrow, ncol))
if opt.n_channels == 1:
    env= SnakeEnv(nrow, ncol, colors = 'gray')
else:
    env= SnakeEnv(nrow, ncol, colors = 'rgb')
    
n_train = opt.n_train
n_episodes = opt.n_episodes

imax = 100
res = Results()

loadModel = opt.load
results_filename = 'results{}.pkl'.format(opt.name)
model_filename = 'model{}.h5'.format(opt.name)

    
if loadModel:
    model.load(model_filename)
    with open(results_filename, 'rb') as input:
            res = pickle.load(input)
    epsilon = res.epsilon
    epsilons = np.maximum(np.arange(epsilon,epsilon -0.9, - 0.9/n_train), 0.1)
else:
    epsilons = np.arange(1,0.1, - 0.9/n_train)

# size of memory
N_memory = 50000
for i_train in range(n_train):
    epsilon = epsilons[i_train]
    res.epsilon = epsilon
    print("Training: round {}, epsilon = {}".format(i_train, round(epsilon,2)))
    lengths_i_train = []
    scores_i_train = []
    for i_episode in range(n_episode):
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

    res.memory = res.memory[-N_memory:]
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
