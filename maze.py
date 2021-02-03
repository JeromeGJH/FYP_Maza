import math
import  numpy as np
from agent import *




class Maze:
    def __init__(self, num, learning_rate=0.1, gamma=0.9, memory_size=5000, epsilon = 0.1, shape = (3, 8), AgentP = [], GoalP = []):
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

        Maze.initAgents(self)
        Maze.initGoals(self)



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

    def initWall(self): #bbuild wall in the maze to verify algorithm(not included in the paper)
        a = 0

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

    def train(self):
        gamma = self.gamma
        alpha = self.alpha

        num_agents = self.num_agents
        for step_index1 in range(1000):
            if step_index1 % 100 == 0 and self.epsilon > 0:
                self.epsilon -= 0.01
            for agent_index in range(num_agents):
                a = self.agent[agent_index]
                a.position = Maze.initialP[agent_index]
            while True:
                for agent_index in range(num_agents):
                    a = self.agent[agent_index]
                    index1 = self.coordinateToIndex(a.position)
                    value1 = self.Q_table[a.name][index1[0]][index1[1]]
                    action1 = self.selectAction(a)

                    a.move(action1)
                    # agent arrive at the new position
                    a.distance = FormACircle.calculateDistance(a)
                    index2 = coordinateToIndex(a.position)
                    value2 = self.Q_table[a.name][index2[0]][index2[1]]
                    action2 = np.argmax(self.Q_table[a.name][index2[0]][index2[1]])
                    if a.ifOnCircle is True:  # the original position is on the circle
                        # agent moves on the circle
                        if FormACircle.radius + 1 >= a.distance >= FormACircle.radius - 1:
                            self.Q_table[a.name][index1[0]][index1[1]][action1] = value1[action1] + alpha * (
                                        gamma * value2[action2] - value1[action1])
                        # agent leaves the circle
                        else:
                            a.ifOnCircle = False
                            self.onCircle_count -= 1
                            self.Q_table[a.name][index1[0]][index1[1]][action1] = value1[action1] + alpha * (
                                        gamma * value2[action2] - 10 - value1[action1])
                    else:  # the original position is not on the circle
                        # the first time agent moves onto the circle
                        if FormACircle.radius + 1 >= a.distance >= FormACircle.radius - 1:
                            a.ifOnCircle = True
                            self.onCircle_count += 1
                            self.Q_table[a.name][index1[0]][index1[1]][action1] = value1[action1] + alpha * (
                                        gamma * value2[action2] + 10 - value1[action1])
                        # agent is still outside the circle
                        else:
                            self.Q_table[a.name][index1[0]][index1[1]][action1] = value1[action1] + alpha * (
                                        gamma * value2[action2] - 1 - value1[action1])
                if self.onCircle_count == num_agents:
                    break

    def getPayoff(self, agent):















if __name__ == '__main__':
    shape = (3, 8)
    num = 2

    maze = Maze(num=2, shape=shape)
    maze
