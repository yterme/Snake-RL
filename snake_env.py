#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 17:09:16 2018

@author: yannickterme
"""



import curses
from random import randint
import numpy as np

class SnakeEnv:
    UP, LEFT, RIGHT, DOWN = 0, 1, 2, 3
    DEATH_REWARD, APPLE_REWARD, LIFE_REWARD = -1, 1, 0
    
    def __init__(self,nrow, ncol, colors = 'gray',render = False):
        self.nrow = nrow
        self.ncol = ncol
        self.colors = colors
        if colors =='rgb':
            self.HEAD_COLOR, self.SNAKE_COLOR, self.FOOD_COLOR, self.BLANK_COLOR =\
                    [255, 0, 0],[200, 100, 100], [100, 0, 0], [0,0,0]
        elif colors=='gray':       
            self.HEAD_COLOR, self.SNAKE_COLOR, self.FOOD_COLOR, self.BLANK_COLOR = [255], [150], [100], [0]
            
        food = [randint(0,nrow-1), randint(0,ncol-1)]
        self.food = food
        while True:                                              # First food co-ordinates
            snake= [[randint(0,nrow-1), randint(1,ncol-2)]]#[[4,10], [4,9], [4,8]]   
            snake = [snake[0], [snake[0][0], snake[0][1]-1]]    # Initial snake co-ordinates
            if not(food in snake):
                 break
        self.snake = snake
        
        grid = [[self.BLANK_COLOR for _ in range(ncol)] for _ in range(nrow)]
        grid = np.array(grid, dtype = float)
        for i,s in enumerate(self.snake):
            if i==0:
                grid[s[0],s[1]] = self.HEAD_COLOR
            else:
                grid[s[0],s[1]] = self.SNAKE_COLOR
        grid[self.food[0], self.food[1]] = self.FOOD_COLOR
        self.grid = grid
        self.score = 0
        self.actions = [0, 1, 2, 3] #[0, -1], [-1,0], [1,0]]        
        self.render = render

            
    def reset(self):
        self.__init__(self.nrow, self.ncol, self.render)
        return self.grid
    
    def step(self,action):
        
        
        assert(action in self.actions)
        reward = 0
        
        self.snake.insert(0, [self.snake[0][0] + (action == SnakeEnv.DOWN and 1) + (action == SnakeEnv.UP and -1),\
                         self.snake[0][1] + (action == SnakeEnv.LEFT and -1) + (action == SnakeEnv.RIGHT and 1)])
        if self.snake[0][0] == -1 or self.snake[0][0] == self.nrow or \
            self.snake[0][1] == -1 or self.snake[0][1] == self.ncol: # snake hits a wall
            self.score += SnakeEnv.DEATH_REWARD
            reward += SnakeEnv.DEATH_REWARD
            return self.grid, reward, True
        elif self.snake[0] in self.snake[1:]: # snake bites itself
            self.score += SnakeEnv.DEATH_REWARD
            return self.grid, reward, True
        elif self.snake[0] == self.food: # When snake eats the food
            food = []
            self.score += SnakeEnv.APPLE_REWARD
            reward += SnakeEnv.APPLE_REWARD
            while food == []:
                food = [randint(0, self.nrow-1), randint(0, self.ncol-1)] # Calculating next food's coordinates
                if food in self.snake: 
                    food = []
            
            self.food = food
            
            if len(self.snake)==self.nrow*self.ncol: # the snake covers the whole grid, game is won
                score += 1000
                reward += 1000

        else:
            # [1] If it does not eat the food
            last = self.snake.pop()
            self.grid[last[0], last[1]]= self.BLANK_COLOR
  
        for i,s in enumerate(self.snake):
            if i==0:
                self.grid[s[0],s[1]] = self.HEAD_COLOR
            else:
                self.grid[s[0],s[1]] = self.SNAKE_COLOR
        self.grid[self.food[0], self.food[1]] = self.FOOD_COLOR
        
        done = False
        reward+= SnakeEnv.LIFE_REWARD
        return self.grid.copy(), reward, done       
        
    
    def close(self):
        if self.render:
            curses.endwin()
