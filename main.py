import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint
from collections import deque
import random
import tensorflow as tf
import gym
import A2C



print("Program start:")

##Hyperparameters
episodes = 5000 #number of games


#cb = TensorBoard()


class DQNAgent:

    #Create agent
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        #Previous experiences
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 #discount rate
        self.epsilon = 1.0 #exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _customLoss(self, target, pred):
        return 0

    #Build NN
    def _build_model(self):
        model = Sequential()

        # 1st layer with input size of 4 and 24 nodes
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))

        # 2nd layer with 24 nodes
        model.add(Dense(24, activation='relu'))

        # Output layer with 2 nodes for actions (left, right)
        model.add(Dense(self.action_size, activation='linear'))

        model.summary()

        # Compile model
        model.compile(loss='mse', optimizer=optimizers.Adam(self.learning_rate),
                      metrics=['accuracy'])

        return model

    #Save previous experiences
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        #Explore
        if np.random.rand() <= self.epsilon:
            #return random action
            return random.randrange(self.action_size)
        #Compute action probabilities
        act_values = self.model.predict(state)

        return np.argmax(act_values[0]) #returns highest action


    #Method that trains the neural net with experiences in the memory
    def replay(self, batch_size):

        #Fetch random memories
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == '__main__':


    #init gym
    env = gym.make('CartPole-v0')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    #Deep reinforcement
    #agent = DQNAgent(state_size, action_size)

    #A2C agent
    agent2 = A2C.A2CAgent(state_size, action_size)

    #Iterate game
    for e in range(episodes):

        #reset state
        state = env.reset()
        state = np.reshape(state, [1, 4])

        for time_t in range(500):
            #Rendering
            env.render()

            #decide action
            #action = agent.act(state)
            action = agent2.get_action(state)

            #advance to next state
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            #Save state, action, reward and done
            #agent.remember(state, action, reward, next_state, done)


            #A2C
            agent2.train_model(state, action, reward, next_state, done)

            #Next state to current state
            state = next_state

            #when game ends
            if done:

                print("episode: {}/{}, score: {}"
                      .format(e, episodes, time_t))
                break

        #train agent
        #if len(agent.memory) > 32:
           # agent.replay(32)
