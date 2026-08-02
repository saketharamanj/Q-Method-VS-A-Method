# Q-Method-VS-A-Method
Two Python scripts that pit reinforcement learning (Q-learning) against classical search (A*) on the same maze-solving task, to see how closely a learned policy can approach a provably optimal path.

absolute.py — the simpler baseline version:
1) Loads a maze from test_new.csv (0 = open path, 1 = wall, 2 = goal) and displays it with Matplotlib
2) Trains a tabular Q-learning agent (dictionary-based Q-table, epsilon-greedy exploration, 200 episodes) to navigate from a fixed start to the goal
3) Runs A* search (Manhattan-distance heuristic, via a priority queue) to find the shortest path on the same maze
4) Compares the two by average steps-per-episode (Q-learning) vs. path length (A*), converts both into a rough "accuracy" percentage relative to the maze's total cell count, and plots Q-learning's steps-per-episode curve against a horizontal line marking A*'s path length

cinema.py — a more developed, reproducible experiment:
1) Procedurally generates a random solvable 25×25 maze using randomized depth-first search, seeded for reproducibility
2) Trains a Q-learning agent with epsilon decay (500 episodes, tuned learning rate/discount), tracking steps and cumulative reward per episode
3) Runs A* on the same maze and times both algorithms
4) Computes a Q-learning optimality score: how close the agent's average steps (over its final episodes) come to A*'s optimal path length
5) Logs training data to q_training_log.csv and a run summary (maze size, start/goal, path lengths, runtimes, optimality %) to summary.json
6) Produces two saved plots: a convergence chart (steps-per-episode with a rolling average, against A*'s optimal path as a reference line) and a color-coded maze visualization overlaying the A* path and the learned Q-path to show where they agree and diverge

Tech stack: Python, NumPy, pandas, Matplotlib, heapq (for A*)
