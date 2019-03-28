#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 21:30:41 2018

@author: anchuzhu
"""
import random
import math
import pygame
import numpy as np
import sys
#enviroment
#(ball_x, ball_y, velocity_x, velocity_y, paddle_y)


paddle_height = 0.2
action_list = [0, 0.04,  -0.04]  #nothing, down, up
REWARD = 1
PENALTY = -1
epoch = 20000

C = 5 #constant for learn rate, need us to choose

class MDP:
    def __init__(self,ball_x,ball_y,velocity_x,velocity_y,paddle_y,reward):
        self.ball_x = ball_x
        self.ball_y = ball_y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.paddle_y =paddle_y
        self.reward = reward
        

    def update(self,action):
        self.paddle_y += action_list[action]
        self.paddle_y = max(0, min(1 -paddle_height, self.paddle_y))
        self.ball_x += self.velocity_x
        self.ball_y += self.velocity_y
        #top
        if self.ball_y < 0:
            self.ball_y = -self.ball_y  
            self.velocity_y = -self.velocity_y
            #bot
        if self.ball_y > 1:
            self.ball_y = 2 - self.ball_y
            self.velocity_y = -self.velocity_y
            #left
        if self.ball_x < 0:
            self.ball_x = -self.ball_x
            self.velocity_x = -self.velocity_x
        if self.ball_x < 1: 
            return 0
        #bounce
        elif self.ball_x > 1 and (self.ball_y < self.paddle_y or self.ball_y > self.paddle_y+0.2):
            return PENALTY
        else:
            self.ball_x = 2 * 1 - self.ball_x
            old_velocity_x = self.velocity_x
            self.velocity_x = -self.velocity_x + random.uniform(-0.015,0.015)
            while abs(self.velocity_x) <= 0.03:
                self.velocity_x =  -old_velocity_x + random.uniform(-0.015, 0.015)
            self.velocity_y = self.velocity_y + random.uniform(-0.03, 0.03)
            if abs(self.velocity_x) > 1:
                print('the velocity_x increases above 1')
            if abs(self.velocity_y) > 1:
                print('the velocity_y increases above 1')
            #print('aaaaaaaaa')
            return REWARD
        
        
        
    def convert(self):
        ball_x_dis = math.floor(self.ball_x * 12)
        ball_y_dis = math.floor(self.ball_y * 12)
        if self.velocity_x > 0 :
            vx_dis = 1
        else:
            vx_dis = -1
        if abs(self.velocity_y) <0.015:
            vy_dis = 0
        elif self.velocity_y > 0.015: 
            vy_dis = 1
        else:
            vy_dis = -1
            
        if self.paddle_y == 1 - paddle_height:
            discrete_paddle = 11
        else:
            discrete_paddle = math.floor(12 * self.paddle_y / (1 - paddle_height))
        
        
        return (ball_x_dis,ball_y_dis,vx_dis,vy_dis,discrete_paddle)
    
class q_learn:
    def __init__(self):
        self.Q = {}
        self.N = {}
        self.actions = [0,1,2]
        
        
    def get_q(self,state,action):
        return self.Q.get((state,action),0)
    def set_q(self,state,action,value):
        self.Q[(state,action)] = value
    
    def get_action(self,epoch,state):
        elipson = 0.05#max(0.1,1-math.log2(epoch*10)/20)
        if random.random() < elipson:
            random_act = random.choice(self.actions)
            value = self.get_q(state,random_act)
            return random_act, value
        else:
            temp_q = []
            for action in self.actions:
                temp_q.append(self.get_q(state,action))
            MAX = max(temp_q)
            #print(MAX)
            return self.actions[temp_q.index(MAX)], MAX
        
def train_q(agent):
    print('Start Training!')
    total = 0
    pre_tot = 0
    for i in range(1,epoch+1):
        bounce = 0
        state = MDP(0.5, 0.5, 0.03, 0.01, 0.5 - paddle_height / 2,0) 
        curr_state = state.convert()
        while True:
            #print('train')
            
            
            action, old_value = agent.get_action(i,curr_state)
            
            if (curr_state,action) in agent.N:
                agent.N[(curr_state,action)] += 1
            else:
                agent.N[(curr_state,action)] = 0
           
            reward = state.update(action)
            #print(reward)
            new_state = state.convert()
            
            temp_value = []
            '''
            for a in agent.actions:
                temp_value.append(agent.get_q(new_state,action))
            future_value = max(temp_value)
            '''
            _,future_value =agent.get_action(i,new_state)
            #print(future_value)
            
            
            
                
            n = agent.N[(curr_state,action)]
            learn_rate = C / (C + n)
            #print(learn_rate)
            if old_value is None:
                agent.set_q(curr_state,action,reward)
            else:
                #print(curr_state,old_value,reward,future_value)
                new_value = (1-learn_rate)*old_value + learn_rate*(reward +  0.8*(future_value)) #discount factor 0.8?
                #print(new_value,old_value)
                #new_value = old_value + learn_rate(reward +  0.7*(future_value)-old_value)
                #print(curr_state,old_value,new_state,future_value,action,new_value)
                agent.set_q(curr_state,action,new_value)
    #                print(new_value)
            if reward == -1:
                break
            bounce += reward
            curr_state = new_state
        total += bounce
        if i%1000 == 0:
            print('loop',i)
            print('now average bounce in this 1000 round is', (total-pre_tot)/1000)
            print('bounce=',total-pre_tot)
            print()
            pre_tot = total

def test_q(agent):
    print('Start Testing')
    total = 0
    for i in range(epoch):
        bounce = 0 
        state = MDP(0.5, 0.5, 0.03, 0.01, 0.5 - paddle_height / 2,0) 
        curr_state = state.convert()
        while True:
            action, _= agent.get_action(i,curr_state)
            reward = state.update(action)
            new_state = state.convert()
            if reward == -1:
                break
            bounce += reward
            curr_state = new_state
        total += bounce
    print('Average bounce:', total/epoch)
    


    
WHITE = (255, 255, 255)
BLACK = (0,0,0)
BALL_COLOR = (77,166,255)
PAD_COLOR = (255, 51, 153)
BACKGROUND_COLOR = (223, 239, 213)
#SCORE_COLOR = (25, 118, 210)

WIDTH = 500
HEIGHT = 500

BALL_RADIUS = 8
PAD_WIDTH = 8
PAD_HEIGHT = HEIGHT * 0.2
GAME_FPS = 30


#pygame.draw.rect(screen, COLOUR, [x, y, width, height], line_thickness)
'''
pygame.init()
fps = pygame.time.Clock()

window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pong')

for event in pygame.event.get():
    if event.type == pygame.QUIT:
        pygame.quit()
        sys.exit()
canvas = pygame.display.set_mode((WIDTH, HEIGHT))
canvas.fill(BACKGROUND_COLOR)

pygame.draw.rect(canvas, PAD_COLOR, [200,10,200,10], 0)
pygame.display.update()
fps.tick(GAME_FPS)
ball_pos = (int(6 * WIDTH) , int(6 * HEIGHT))
'''   



def draw(canvas,agent,state,player_y):
    canvas.fill(BACKGROUND_COLOR)
        
    ball = (int(state.ball_x * WIDTH) , int(state.ball_y * HEIGHT))
    pygame.draw.circle(canvas, BALL_COLOR, ball, BALL_RADIUS, 0)
    paddle = (WIDTH+1-PAD_WIDTH,int(state.paddle_y*WIDTH),PAD_WIDTH,PAD_HEIGHT)
    pygame.draw.rect(canvas, PAD_COLOR,paddle,  0)
    wall = (0 , 0 ,PAD_WIDTH,500)
    pygame.draw.rect(canvas, PAD_COLOR,wall,  0)
    #player_y = (pygame.mouse.get_pos())[1]
    #player =(0, int(player_y),PAD_WIDTH,PAD_HEIGHT)
    #pygame.draw.rect(canvas, PAD_COLOR,player,  0)
    
def pong(agent):
        
    pygame.init()
    fps = pygame.time.Clock()
    canvas = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('CS440-MP4-Q_Learning')
   
    total = 0
    for i in range(epoch):
        bounce = 0 
        state = MDP(0.5, 0.5, 0.03, 0.01, 0.5 - paddle_height / 2,0) 
        curr_state = state.convert()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            
            action, _= agent.get_action(i,curr_state)
            #print(player_y,state.paddle_y)
            reward = state.update(action)
            new_state = state.convert()
            if reward == -1:
                break
            
            draw(canvas,agent,state,player_y)
            pygame.display.update()
            fps.tick(GAME_FPS)
        
            bounce += reward
            curr_state = new_state
            
        total += bounce
    fps.tick(GAME_FPS)
    print(total)
  
agent = q_learn()
train_q(agent)
#test(agent)
pong(agent) 

        