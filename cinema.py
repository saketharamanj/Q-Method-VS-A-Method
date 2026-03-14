import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import json
import os 
from heapq import heappush, heappop
from collections import defaultdict


SEED = 42
np.random.seed(SEED)
random.seed(SEED)

FIGURES_DIR = "figs"
if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(i) for i in obj)
    return obj

def generate_solvable_maze(rows, cols):
    grid = np.ones((rows, cols), dtype=int)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    stack = [(1, 1)] 
    grid[1, 1] = 0 
    while stack:
        r, c = stack[-1]
        unvisited_neighbors = []
        for dr, dc in directions:
            nr, nc = r + 2 * dr, c + 2 * dc 
            if 0 < nr < rows - 1 and 0 < nc < cols - 1 and grid[nr, nc] == 1:
                unvisited_neighbors.append((nr, nc, dr, dc))
        
        if unvisited_neighbors:
            nr, nc, dr, dc = random.choice(unvisited_neighbors)
            grid[r + dr, c + dc] = 0
            grid[nr, nc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()
            
    final_maze = np.copy(grid)
    start_pos = (1, 1)
    goal_pos = (rows - 2, cols - 2)
    final_maze[start_pos] = 0 
    final_maze[goal_pos] = 2 

    return final_maze, start_pos, goal_pos

MAZE_ROWS, MAZE_COLS = 25, 25
maze, start_pos, goal_pos = generate_solvable_maze(MAZE_ROWS, MAZE_COLS)
MAZE_FILE_NAME = "Generated Maze (25x25)" 

print(f"Maze loaded successfully: {MAZE_FILE_NAME} (Start: {start_pos}, Goal: {goal_pos})")
print("Maze shape:", maze.shape)

class MazeEnv:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.start = start
        self.state = start
        self.goal = goal
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
            return self.state, -5, False 

        self.state = (r, c)
        if self.state == self.goal:
            return self.state, 10, True
        
        return self.state, -1, False

def q_learning(env, episodes=500, alpha=0.9, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, decay_rate=0.005, max_steps=1000):
    Q = defaultdict(float) 
    steps_per_episode = []
    cumulative_rewards = []
    
    for ep in range(episodes):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-decay_rate * ep)
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        while not done and steps < max_steps: 
            
            if random.random() < epsilon:
                action = random.choice(env.actions)
            else:
                q_vals = [Q[(state, a)] for a in env.actions]
                if len(set(q_vals)) <= 1: 
                    action = random.choice(env.actions)
                else:
                    action = env.actions[np.argmax(q_vals)]

            next_state, reward, done = env.step(action)
            
            old_q = Q[(state, action)]
            next_max_q = max([Q[(next_state, a)] for a in env.actions])
            Q[(state, action)] = old_q + alpha * (reward + gamma * next_max_q - old_q)

            state = next_state
            steps += 1
            total_reward += reward

        steps_per_episode.append(steps)
        cumulative_rewards.append(total_reward)

    return Q, steps_per_episode, cumulative_rewards

def a_star(maze, start, goal):
    rows, cols = maze.shape
    h = lambda pos: abs(goal[0]-pos[0]) + abs(goal[1]-pos[1]) 
    
    open_list = []
    heappush(open_list, (h(start), start)) 
    came_from = {}
    g_score = {start: 0} 
    
    while open_list:
        _, current = heappop(open_list)
        
        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = came_from.get(current)
            path.append(start)
            path.reverse()
            return path

        r, c = current
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]: 
            nr, nc = r+dr, c+dc
            neighbor = (nr, nc)
            
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] != 1:
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + h(neighbor)
                    heappush(open_list, (f_score, neighbor))
    return None


env = MazeEnv(maze, start=start_pos, goal=goal_pos)
start, goal = env.start, env.goal
EPISODES = 500

start_time_a = time.time() 
a_star_path = a_star(maze, start, goal)
a_star_runtime = time.time() - start_time_a
a_star_steps = len(a_star_path) - 1 if a_star_path else np.inf 

start_time_q = time.time() 
Q, q_steps, q_rewards = q_learning(env, episodes=EPISODES)
q_runtime = time.time() - start_time_q

LAST_N = max(1, len(q_steps) // 10)
q_learned_steps = np.mean(q_steps[-LAST_N:])

if a_star_steps == np.inf:
    optimal_path_pct = 0
    analysis_text = "A* could not find a path. Q-Learning optimality cannot be calculated."
else:
    optimality = a_star_steps / q_learned_steps if q_learned_steps > 0 else 0
    optimal_path_pct = min(100.0, optimality * 100) 
    
    analysis_text = (
        f"--- Performance Summary and Analysis ---\n"
        f"Maze Source: {MAZE_FILE_NAME}\n"
        f"A* Optimal Path Length: {a_star_steps}\n"
        f"Q-Learning Avg Steps (Last {LAST_N} Episodes): {q_learned_steps:.2f}\n"
        f"Q-Learning Optimality: {optimal_path_pct:.2f}%\n\n"
        f"A* Runtime: {a_star_runtime:.4f}s\n"
        f"Q-Learning Training Runtime: {q_runtime:.4f}s"
    )

print("\n" + analysis_text)


log_data = pd.DataFrame({
    'Episode': range(EPISODES),
    'Steps_Taken': q_steps,
    'Cumulative_Reward': q_rewards
})
log_data['Steps_MA_50'] = log_data['Steps_Taken'].rolling(window=50).mean()
log_data['Reward_MA_50'] = log_data['Cumulative_Reward'].rolling(window=50).mean()
log_data.to_csv("q_training_log.csv", index=False)
print("Saved Q-Learning training log to q_training_log.csv")

summary_data = {
    "maze_source": MAZE_FILE_NAME,
    "maze_shape": convert_to_serializable(maze.shape),
    "start_position": convert_to_serializable(start),
    "goal_position": convert_to_serializable(goal),
    "a_star_steps": convert_to_serializable(a_star_steps),
    "a_star_runtime_s": convert_to_serializable(a_star_runtime),
    "q_learning_episodes": EPISODES,
    "q_learning_avg_steps_learned": convert_to_serializable(q_learned_steps),
    "q_learning_runtime_s": convert_to_serializable(q_runtime),
    "optimality_pct": convert_to_serializable(optimal_path_pct)
}
with open("summary.json", 'w') as f:
    json.dump(summary_data, f, indent=4)
print("Saved summary data to summary.json")

def get_q_path(env, Q, max_steps):
    state = env.start
    path = [state]
    for _ in range(max_steps): 
        if state == env.goal: break
        q_vals = [Q[(state, a)] for a in env.actions]
        if not q_vals: break
        action = env.actions[np.argmax(q_vals)]
        r, c = state
        if action == 'up': r -= 1
        elif action == 'down': r += 1
        elif action == 'left': c -= 1
        elif action == 'right': c += 1
        if 0 <= r < env.n_rows and 0 <= c < env.n_cols and env.maze[r, c] != 1:
            state = (r, c)
            path.append(state)
        else:
            break 
    return path

q_path = get_q_path(env, Q, max_steps=maze.size * 2)

plt.figure(figsize=(10, 6))
plt.plot(log_data['Steps_Taken'], alpha=0.3, label="Steps per Episode")
plt.plot(log_data['Steps_MA_50'], color='blue', label="Steps 50-Episode MA") 
plt.axhline(y=a_star_steps, color='r', linestyle='--', label=f"A* Optimal Path ({a_star_steps} Steps)")
plt.xlabel("Episode")
plt.ylabel("Steps Taken")
plt.title(f"Q-Learning Convergence to A* Optimum (Optimality: {optimal_path_pct:.2f}%)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig(f"{FIGURES_DIR}/steps_taken_vs_episode.png")
plt.show()

path_map = np.copy(maze).astype(float)
path_map[path_map == 2] = 0.8 
path_map[start] = 0.6 
if a_star_path:
    for r, c in a_star_path:
        if path_map[r, c] == 0:
            path_map[r, c] = 0.4 
for r, c in q_path:
    if path_map[r, c] == 0.4: 
        path_map[r, c] = 0.5
    elif path_map[r, c] == 0: 
        path_map[r, c] = 0.2

plt.figure(figsize=(8, 8))
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap_colors = {0.0: 'white', 0.2: 'red', 0.4: 'blue', 0.5: 'purple', 0.6: 'lime', 0.8: 'yellow', 1.0: 'black'}
keys = sorted(cmap_colors.keys())
cmap = ListedColormap([cmap_colors[k] for k in keys])
norm = BoundaryNorm(keys + [1.1], cmap.N)

plt.imshow(path_map, cmap=cmap, norm=norm)
plt.title(f"Learned Q-Path vs A* Path on {MAZE_FILE_NAME}\nStart: {start}, Goal: {goal}")
plt.colorbar(ticks=[0.2, 0.4, 0.5, 0.6, 0.8, 1.0], label='Color Key: Red (Q), Blue (A*), Purple (Overlap)')
plt.savefig(f"{FIGURES_DIR}/path_overlay.png")
plt.show()

print("\nCode execution complete with a generated solvable maze. Results are saved.")
