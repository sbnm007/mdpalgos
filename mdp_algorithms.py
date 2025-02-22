import numpy as np

ACTIONS = [(1,0), (-1,0), (0,1), (0,-1)]  # Right, Left, Down, Up

def build_mdp_model(maze, goal, step_reward=0, goal_reward=10):
    """
    Builds a dictionary-based MDP model:
      - states: all (x,y) cells that are 0 (open)
      - transitions: from each state, possible next states given an action
      - rewards: R(s,a,s') is step_reward for each move, and goal_reward for the goal.
    """
    states = []
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y][x] == 0:
                states.append((x, y))

    transitions = {}
    rewards = {}
    for s in states:
        transitions[s] = {}
        x, y = s
        for a in ACTIONS:
            nx, ny = x + a[0], y + a[1]
            if (nx, ny) in states:
                # Valid transition
                transitions[s][a] = [(1.0, (nx, ny))]
                # Reward
                if (nx, ny) == goal:
                    rewards[(s, a, (nx, ny))] = goal_reward
                else:
                    rewards[(s, a, (nx, ny))] = step_reward
            else:
                # If invalid, remain in same state or no transitions
                transitions[s][a] = [(1.0, s)]
                # Typically no reward for hitting a wall
                rewards[(s, a, s)] = step_reward
    return states, transitions, rewards

def value_iteration(states, transitions, rewards, gamma=0.9, theta=1e-4):
    """
    Performs Value Iteration on the given MDP model.
    :param states: list of all states
    :param transitions: transitions[s][a] = [(prob, next_state), ...]
    :param rewards: rewards[(s,a,s')]
    :param gamma: discount factor
    :param theta: convergence threshold
    :return: dictionary of values V[s], and an optimal policy (mapping from s to best action)
    """
    V = {s: 0.0 for s in states}  # initialize values
    policy = {s: None for s in states}

    while True:
        delta = 0
        for s in states:
            v = V[s]
            # Bellman update
            best_value = float('-inf')
            best_action = None
            for a in transitions[s]:
                q_sa = 0
                for (prob, s_next) in transitions[s][a]:
                    r = rewards[(s, a, s_next)]
                    q_sa += prob * (r + gamma * V[s_next])
                if q_sa > best_value:
                    best_value = q_sa
                    best_action = a
            V[s] = best_value
            policy[s] = best_action
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V, policy

def policy_iteration(states, transitions, rewards, gamma=0.9, theta=1e-4):
    """
    Performs Policy Iteration on the given MDP model.
    :return: dictionary of values V[s], and an optimal policy
    """
    # 1. Initialize arbitrary policy, ensuring states with no valid actions get None
    policy = {
        s: np.random.choice(list(transitions[s].keys())) if transitions[s] else None
        for s in states
    }
    V = {s: 0.0 for s in states}

    def policy_evaluation(policy, V):
        while True:
            delta = 0
            for s in states:
                v = V[s]
                a = policy[s]
                if a is None:  # Skip states with no valid actions
                    continue
                new_v = 0
                for (prob, s_next) in transitions[s][a]:
                    r = rewards.get((s, a, s_next), 0)
                    new_v += prob * (r + gamma * V[s_next])
                V[s] = new_v
                delta = max(delta, abs(v - new_v))
            if delta < theta:
                break

    while True:
        # 2. Policy Evaluation
        policy_evaluation(policy, V)

        # 3. Policy Improvement
        policy_stable = True
        for s in states:
            old_action = policy[s]
            best_action = None
            best_value = float('-inf')

            if transitions[s]:  # Ensure transitions exist
                for a in transitions[s]:
                    q_sa = 0
                    for (prob, s_next) in transitions[s][a]:
                        r = rewards.get((s, a, s_next), 0)
                        q_sa += prob * (r + gamma * V[s_next])
                    if q_sa > best_value:
                        best_value = q_sa
                        best_action = a

                policy[s] = best_action
                if best_action != old_action:
                    policy_stable = False

        if policy_stable:
            break

    return V, policy

