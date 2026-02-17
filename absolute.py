import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from heapq import heappush, heappop
maze = pd.read_csv("test_new.csv", header=None).apply(pd.to_numeric, errors='coerce').values
print("Maze shape:", maze.shape)

plt.imshow(maze, cmap='gray_r')
plt.title("Maze Layout")
plt.show()
class MazeEnv:
    def __init__(self, maze, start=(0,0)):
        self.maze = maze
        self.start = start
        self.state = start
        self.goal = tuple(np.argwhere(maze == 2)[0])
        self.actions = ['up', 'down', 'left', 'right']
        self.n_rows, self.n_cols = maze.shape

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        r, c = self.state
        if action == 'up': r -= 1
        elif action == 'down': r += 1
        elif action == 'left': c -= 1
        elif action == 'right': c += 1

        if r < 0 or c < 0 or r >= self.n_rows or c >= self.n_cols or self.maze[r, c] == 1:
            return self.state, -1, False

        self.state = (r, c)
        if self.state == self.goal:
            return self.state, 10, True  
        return self.state, -0.1, False
def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2):
    Q = {}
    steps_per_episode = []

    def get_Q(state, action):
        return Q.get((state, action), 0.0)

    for ep in range(episodes):
        state = env.reset()
        done = False
        steps = 0

        while not done:
            if random.random() < epsilon:
                action = random.choice(env.actions)
            else:
                q_vals = [get_Q(state, a) for a in env.actions]
                action = env.actions[np.argmax(q_vals)]

            next_state, reward, done = env.step(action)
            old_q = get_Q(state, action)
            next_max_q = max([get_Q(next_state, a) for a in env.actions])
            Q[(state, action)] = old_q + alpha * (reward + gamma * next_max_q - old_q)

            state = next_state
            steps += 1

        steps_per_episode.append(steps)

    return Q, steps_per_episode
def a_star(maze, start, goal):
    rows, cols = maze.shape
    open_list = []
    heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(goal[0]-start[0]) + abs(goal[1]-start[1])}
    visited = set()

    while open_list:
        _, current = heappop(open_list)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        visited.add(current)
        r, c = current
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            neighbor = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] != 1:
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + abs(goal[0]-nr) + abs(goal[1]-nc)
                    heappush(open_list, (f_score[neighbor], neighbor))
    return None
env = MazeEnv(maze)
start, goal = env.start, env.goal

Q, q_steps = q_learning(env, episodes=200)

a_star_path = a_star(maze, start, goal)

q_avg_steps = np.mean(q_steps)
a_star_steps = len(a_star_path) if a_star_path else np.inf

print(f"Average steps per Q-learning episode: {q_avg_steps:.2f}")
print(f"A* path length: {a_star_steps}")

max_possible = maze.shape[0] * maze.shape[1]
q_accuracy = (1 - q_avg_steps / max_possible) * 100
a_accuracy = (1 - a_star_steps / max_possible) * 100

print(f"Q-learning Accuracy: {q_accuracy:.2f}%")
print(f"A* Accuracy: {a_accuracy:.2f}%")
plt.figure(figsize=(8,5))
plt.plot(q_steps, label="Q-Learning Steps per Episode")
plt.axhline(y=a_star_steps, color='r', linestyle='--', label="A* Path Length")
plt.xlabel("Episode")
plt.ylabel("Steps Taken")
plt.title("Q-Learning vs A* Maze Solver")
plt.legend()
plt.show()
