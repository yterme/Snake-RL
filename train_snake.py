from snake_env import SnakeEnv
import numpy as np
from dicts import int2tuple, tuple2int
from initialize import initialize_graph
from actions import get_action, get_current_direction

from DQNetwork import DQNetwork

def test(env, model, episode_count = 100, imax = 100):
    lengths = []
    scores = []
    for i_episode in range(episode_count):
        #print("Episode",str(i_episode))
        grid = env.reset()
        i=0
        done = False
        epsilon = 0

        while i <imax:
            grid= grid.reshape((1,1, env.nrow, env.ncol))
            action = np.argmax(model.predict(grid))
            grid, reward, done = env.step(action)

            i+=1
            if done:
                break
        lengths.append(i)
        scores.append(env.score)
    mean_length = np.mean(lengths)
    mean_score = np.mean(scores)
#print("Mean length of episode:", np.mean(lengths[i_train]))
#print("Mean score:", np.mean(scores[i_train]))

    return(mean_length, mean_score)


nrow, ncol = 5, 5
model = DQNetwork(4, (1,nrow, ncol))

env= SnakeEnv(nrow, ncol, colors = 'gray')

epsilon = 1
#memory = []
n_train = 1000
scores =[[] for i in range(n_train)]
lengths = [[] for i in range(n_train)]
n_batch = 5000
lengths_expl = []
scores_expl = []
epsilons = [1-(1-0.1)*i/(n_train-1) for i in range(n_train)]

memory = []


episode_count = 500
imax = 100

#epsilons = [0.2]*n_train

# size of memory
N = 100000
for i_train in range(n_train):
    #memory = []
    print("\nTraining: round",str(i_train))
    print("Epsilon=",str(round(epsilon,2)))
    for i_episode in range(episode_count):
        #print("Episode",str(i_episode))
        grid = env.reset()
        i=0
        done = False
        epsilon = epsilons[i_train]
        grid= grid.reshape((1,1, env.nrow, env.ncol))
        
        while i <imax:
            i+=1
            grid= grid.reshape((1,1, env.nrow, env.ncol))
            source = grid.copy()
            if epsilon >= np.random.rand():
                action = np.random.randint(4)
            else:
                action = np.argmax(model.predict(source))
            grid, reward, done = env.step(action)
            grid= grid.reshape((1,1, env.nrow, env.ncol))
            observation = {'source':source, 'action':action, \
                           'dest':grid, 'reward':reward,'final':done}
            memory.append(observation)

            if done:
                break
        lengths[i_train].append(i)
        scores[i_train].append(env.score)
    
    memory = memory[-N:]
    #print("Mean length of episode:", np.mean(lengths[i_train]))
    #print("Mean score:", np.mean(scores[i_train]))
    if i_train %5 ==0:
        (l, s) = test(env,model)
        print("Test: mean length {}, mean score {}".format(l,s))
        lengths_expl.append(l)
        scores_expl.append(s)
        
    if len(memory)>=n_batch:
        i_batch = np.random.choice(np.arange(0, len(memory)), n_batch, replace = False)
        batch = [memory[i] for i in i_batch]
    else:
        batch = memory
        
    model.train(batch)
    model.save()
