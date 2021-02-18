import math
import numpy as np
from agent import *


class Maze:
    def __init__(self, num, learning_rate=0.1, gamma=0.9, memory_size=5000, epsilon=0.1, shape=(3, 8), AgentP=[],
                 GoalP=[], WallP=[]):
        self.env = np.array(shape[0], shape[1])
        self.shape = shape
        self.num_agents = num
        self.num_goals = num

        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        self.agent = []
        self.ap = AgentP
        self.gp = GoalP
        self.wp = WallP

        Maze.initAgents(self)
        Maze.initGoals(self)
        Maze.initWall(self)

        self.Q_table = np.zeros((self.num_agents, shape[0], shape[1], 5), dtype=int)

        self.IR_table = np.zeros((self.num_agents, self.num_goals), dtype=int)

        self.min_steps = np.zeros((self.num_agents, 1), dtype=int)

        self.strategy = np.zeros((self.num_agents, 1), dtype=int)

        self.num_arrival = 0

    def coordinateToIndex(self, c):
        return [self.shape[1] - c[1], c[0]]

    def IndexToCoordinate(self, i):
        return [i[1], self.shape[1] - i[0]]

    def initAgents(self):
        for i in range(self.num_agents):
            a = Agent(i, self.ap[i])
            self.agent.append(a)

    def initGoals(self):
        for i in range(self.num_goals):
            goal = Maze.coordinateToIndex(self.gp[i])
            self.env[goal[0]][goal[1]] = 1

    def initWall(self):  # build wall in the maze to verify algorithm(not included in the paper)
        for i in range(len(self.wp)):
            p = Maze.coordinateToIndex(self.wp[i])
            self.env[p[0]][p[1]] = -1

    # get the information of position around the agent没写完
    def getPositionInfo(self, index, direction):
        pInfo = np.array([])
        if 0 in direction:
            np.append(pInfo, self.env[index[0]][index[1]])
        if 1 in direction:
            np.append(pInfo, self.env[index[0]][index[1] - 1])
        if 2 in direction:
            np.append(pInfo, self.env[index[0] - 1][index[1]])
        if 3 in direction:
            np.append(pInfo, self.env[index[0]][index[1] + 1])
        if 4 in direction:
            np.append(pInfo, self.env[index[0] + 1][index[1]])
        return pInfo

    # get the Q-value around the agent
    def getQvalueInfo(self, name, index, direction):
        value = np.array([])
        if 0 in direction:
            np.append(value, np.max(self.Q_table[name][index[0]][index[1]]))
        if 1 in direction:
            np.append(value, np.max(self.Q_table[name][index[0]][index[1] - 1]))
        if 2 in direction:
            np.append(value, np.max(self.Q_table[name][index[0] - 1][index[1]]))
        if 3 in direction:
            np.append(value, np.max(self.Q_table[name][index[0]][index[1] + 1]))
        if 4 in direction:
            np.append(value, np.max(self.Q_table[name][index[0] + 1][index[1]]))
        return value

    # agent is on the border
    def getFeasibleDirection(self, p):
        shape = self.shape
        f_d = []
        if p[0] == 0:
            if p[1] == 0:
                f_d = [0, 2, 3]
            elif p[1] == shape[1]:
                f_d = [0, 3, 4]
            else:
                f_d = [0, 2, 3, 4]
        elif p[0] == shape[0]:
            if p[1] == 0:
                f_d = [0, 1, 2]
            elif p[1] == shape[1]:
                f_d = [0, 1, 4]
            else:
                f_d = [0, 1, 2, 4]
        else:
            if p[1] == 0:
                f_d = [0, 1, 2, 3]
            elif p[1] == shape[1]:
                f_d = [0, 1, 3, 4]
            else:
                f_d = [0, 1, 2, 3, 4]
        return f_d

    # get the feasible action with the maximun payoff
    def getMaxPayoffMove(self, a):
        p = a.position
        index = Maze.coordinateToIndex(self, p)
        name = a.name
        # the agent is not on the circle, move onto the circle

        # feasible directions
        direction = [0, 1, 2, 3, 4]
        if p[0] == 0 or p[0] == 100 or p[1] == 0 or p[1] == 100:
            direction = Maze.getFeasibleDirection(self, p)

        value = Maze.getQvalueInfo(self, name, index, direction)
        pInfo = Maze.getPositionInfo(self, index, direction)
        min_value = np.min(value) - 10
        while True:
            max_value = np.max(value)
            for i in range(len(direction)):
                if value[i] == max_value:
                    if direction[i] == 0:
                        return 0
                    else:
                        if pInfo[i] == 0:
                            return direction[i]
                        else:
                            value[i] = min_value

    def getRandomMove(self, a):
        p = a.position
        index = Maze.coordinateToIndex(self, p)
        name = a.name

        # feasible directions
        direction = [0, 1, 2, 3, 4]
        if p[0] == 0 or p[0] == 100 or p[1] == 0 or p[1] == 100:
            direction = Maze.getFeasibleDirection(self, p)

        while True:
            action = int(np.random.rand() * 5)
            pInfo = Maze.getPositionInfo(self, index, direction)
            if action == 0:
                return 0
            elif pInfo[action] == 0:  # the position has not been occupied
                return action

    # use epsilon-greedy algorithm
    def selectAction(self, a):
        e = np.random.rand()
        if e > self.epsilon:
            return Maze.getMaxPayoffMove(a)
        else:
            return Maze.getRandomMove(a)

    def setIR(self):  # set internal reward
        self.IR_table = 0

    def checkArrival(self, i):  # check whether the agent has arrived at any goal // return number represent goal_index
        a = self.agent[i]
        if a.position not in self.gp:
            return -1
        else:
            return self.gp.index(a.position)

    def train(self):
        gamma = self.gamma
        alpha = self.alpha

        num_agents = self.num_agents

        for step_index1 in range(1000):
            if step_index1 % 100 == 0 and self.epsilon > 0:
                self.epsilon -= 0.01

            for agent_index in range(num_agents):
                a = self.agent[agent_index]
                a.position = self.ap[agent_index]

            steps = 0

            while True:
                for agent_index in range(num_agents):
                    a = self.agent[agent_index]

                    if a.ifArrive is True: # the agent has arrived at the goal
                        continue

                    index1 = self.coordinateToIndex(a.position)
                    value1 = self.Q_table[a.name][index1[0]][index1[1]]
                    action1 = self.selectAction(a)

                    a.move(action1)
                    # agent arrive at the new position

                    index2 = self.coordinateToIndex(a.position)
                    value2 = self.Q_table[a.name][index2[0]][index2[1]]
                    action2 = np.argmax(self.Q_table[a.name][index2[0]][index2[1]])

                    goal_index = self.checkArrival(agent_index)

                    if goal_index >= 0:
                        a.ifArrive = True
                        if steps < self.min_steps[agent_index]:
                            self.min_steps[agent_index] = steps

                    if a.ifArrive is False:
                        self.Q_table[a.name][index1[0]][index1[1]][action1] = value1[action1] + alpha * (
                                gamma * value2[action2] - value1[action1])

                    else:  # the agent arrives at the goal
                        self.Q_table[a.name][index1[0]][index1[1]][action1] = value1[action1] + alpha * (
                                gamma * value2[action2] + self.IR_table[agent_index][goal_index] - value1[action1])

                    steps += 1

                if self.num_arrival == num_agents:
                    break

    def calIR(self, agent):
        a = 0

    def outcome(self):  # output outcome
        for i in range(self.num_agents):
            a = self.agent[i]
            while a.position not in self.gp:
                index = self.coordinateToIndex(a.position)
                action = np.argmax(self.Q_table[a.name][index[0]][index[1]])
                np.append(self.strategy[i], action)
                a.move(action)

        for i in range(self.num_agents):
            print(self.strategy[i])


if __name__ == '__main__':
    shape = (3, 8)
    num = 2
    epsilon = 0.3
    gamma = 0.1
    alpha = 0.1
    AgentPosition = []
    GoalPosition = []
    WallPosition = []

    maze = Maze(num=num, shape=shape)
    maze.train()
    maze.outcome()
