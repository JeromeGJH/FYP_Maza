import numpy as np


class Agent():
    # 1 = left; 2 = up; 3 = right; 4 = down; 0 = stay
    actions = (0, 1, 2, 3, 4)

    def __init__(self, name, p):
        self.position = np.array(p, dtype=int)
        self.distance = 0
        self.direction = []
        self.name = name
        self.reward = 0
        self.goal = 0
        self.strategy = []
        self.ifArrive = False
        self.actions = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]], dtype=int)

    def move(self, action):
        self.position = self.position + self.actions[action]

        # if action == Agent.actions[0]:
        #     self.position[1] += 1
        # elif action == Agent.actions[1]:
        #     self.position[1] -= 1
        # elif action == Agent.actions[2]:
        #     self.position[0] -= 1
        # elif action == Agent.actions[3]:
        #     self.position[0] += 1
