
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv
from gymnasium.error import DependencyNotInstalled
from os import path

# Do not change this class
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
image_path = path.join(path.dirname(gym.__file__), "envs", "toy_text")

class CliffWalking(CliffWalkingEnv):
    def __init__(self, is_hardmode=True, num_cliffs=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_hardmode = is_hardmode

        # Generate random cliff positions
        if self.is_hardmode:
            self.num_cliffs = num_cliffs
            self._cliff = np.zeros(self.shape, dtype=bool)
            self.start_state = (3, 0)
            self.terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
            self.cliff_positions = []
            while len(self.cliff_positions) < self.num_cliffs:
                new_row = np.random.randint(0, 4)
                new_col = np.random.randint(0, 11)
                state = (new_row, new_col)
                if (
                    (state not in self.cliff_positions)
                    and (state != self.start_state)
                    and (state != self.terminal_state)
                ):
                    self._cliff[new_row, new_col] = True
                    if not self.is_valid():
                        self._cliff[new_row, new_col] = False
                        continue
                    self.cliff_positions.append(state)

        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

    def _calculate_transition_prob(self, current, delta):
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_terminated = tuple(new_position) == terminal_state
        return [(1 / 3, new_state, -1, is_terminated)]

    # DFS to check that it's a valid path.
    def is_valid(self):
        frontier, discovered = [], set()
        frontier.append((3, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= self.shape[0] or c_new < 0 or c_new >= self.shape[1]:
                        continue
                    if (r_new, c_new) == self.terminal_state:
                        return True
                    if not self._cliff[r_new][c_new]:
                        frontier.append((r_new, c_new))
        return False

    def step(self, action):
        if action not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid action {action}   must be in [0, 1, 2, 3]")

        if self.is_hardmode:
            match action:
                case 0:
                    action = np.random.choice([0, 1, 3], p=[1 / 3, 1 / 3, 1 / 3])
                case 1:
                    action = np.random.choice([0, 1, 2], p=[1 / 3, 1 / 3, 1 / 3])
                case 2:
                    action = np.random.choice([1, 2, 3], p=[1 / 3, 1 / 3, 1 / 3])
                case 3:
                    action = np.random.choice([0, 2, 3], p=[1 / 3, 1 / 3, 1 / 3])

        return super().step(action)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("CliffWalking - Edited by Audrina & Kian")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.elf_images is None:
            hikers = [
                path.join(image_path, "img/elf_up.png"),
                path.join(image_path, "img/elf_right.png"),
                path.join(image_path, "img/elf_down.png"),
                path.join(image_path, "img/elf_left.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in hikers
            ]
        if self.start_img is None:
            file_name = path.join(image_path, "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(image_path, "img/cookie.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.mountain_bg_img is None:
            bg_imgs = [
                path.join(image_path, "img/mountain_bg1.png"),
                path.join(image_path, "img/mountain_bg2.png"),
            ]
            self.mountain_bg_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in bg_imgs
            ]
        if self.near_cliff_img is None:
            near_cliff_imgs = [
                path.join(image_path, "img/mountain_near-cliff1.png"),
                path.join(image_path, "img/mountain_near-cliff2.png"),
            ]
            self.near_cliff_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in near_cliff_imgs
            ]
        if self.cliff_img is None:
            file_name = path.join(image_path, "img/mountain_cliff.png")
            self.cliff_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        for s in range(self.nS):
            row, col = np.unravel_index(s, self.shape)
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            self.window_surface.blit(self.mountain_bg_img[check_board_mask], pos)

            if self._cliff[row, col]:
                self.window_surface.blit(self.cliff_img, pos)
            if s == self.start_state_index:
                self.window_surface.blit(self.start_img, pos)
            if s == self.nS - 1:
                self.window_surface.blit(self.goal_img, pos)
            if s == self.s:
                elf_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])
                last_action = self.lastaction if self.lastaction is not None else 2
                self.window_surface.blit(self.elf_images[last_action], elf_pos)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )


# Create an environment
def main():
    env = CliffWalking(render_mode="human")
    observation, info = env.reset(seed=30)
    enviroment = enviroment_init()
    values, policy = mdp_algorithm(enviroment, env)
    # Define the maximum number of iterations
    max_iter_number = 1000
    next_state = 36
    counter = 0
    fail = 0
    for i in range(len(policy)):
        print(i, ':', policy[i], end='   ')
    print('\n\n')
    for i in range(len(policy)):
        print(i, ':', values[i], end='   ')

    for __ in range(max_iter_number):
        # TODO: Implement the agent policy here
        # Note: .sample() is used to sample random action from the environment's action space

        # Choose an action (Replace this random action with your agent's policy)
        action = policy[next_state]

        # Perform the action and receive feedback from the environment
        next_state, reward, done, truncated, info = env.step(action)

        if next_state == 47:
            counter += 1

        if reward == -100:
            fail += 1

        # print(next_state)

        if done or truncated:
            observation, info = env.reset()

    # Close the environment
    print(counter)
    print(fail)

    env.close()

def actions(s, a):
    r = s // 12
    c = s % 12

    action = []
    if a == 0:
        action.append(0)
        if c != 0:
            action.append(3)
        else:
            action.append(-1)

        if c != 11:
            action.append(1)
        else:
            action.append(-1)

    if a == 1:
        action.append(1)
        if r != 0:
            action.append(0)
        else:
            action.append(-1)

        if r != 3:
            action.append(2)
        else:
            action.append(-1)

    if a == 2:
        action.append(2)
        if c != 0:
            action.append(3)
        else:
            action.append(-1)

        if c != 11:
            action.append(1)
        else:
            action.append(-1)

    if a == 3:
        action.append(3)
        if r != 0:
            action.append(0)
        else:
            action.append(-1)

        if r != 3:
            action.append(2)
        else:
            action.append(-1)

    return action


def possible_actions(s):
    r = s // 12
    c = s % 12
    action = []

    if r != 0:
        action.append(0)
    if r != 3:
        action.append(2)
    if c != 0:
        action.append(3)
    if c != 11:
        action.append(1)

    return action

def next_state(s, a):
    if a == 0:
        return s - 12
    if a == 1:
        return s + 1
    if a == 2:
        return s + 12
    if a == 3:
        return s - 1
    if a == -1:
        return s


def update_values(enviroment, env, policy, gamma, threshold=0.0001):
    values = [0 for i in range(env.observation_space.n)]
    q_values = {}
    for state in range(48):
        q_values[state] = {}
        for ac in possible_actions(state):
            q_values[state][ac] = 0

    cliff = [u[0] * 12 + u[1] for u in env.cliff_positions]
    t = 0
    while (t < 1000):
        distance = 0
        last_val = values.copy()
        for state in range(env.observation_space.n):
            if (state == 47):
                mx_q = 4000
            elif (state in cliff):
                mx_q = -100
            else:
                for action in possible_actions(state):
                    q_values[state][action] = 0
                    sum = 0
                    for probability, reward, nxt_state in enviroment[state][action]:
                        q_values[state][action] += probability * (reward + gamma * values[nxt_state])
                    mx_q = q_values[state][policy[state]]

            values[state] = mx_q

        t += 1

    return q_values, values

def enviroment_init():
    enviroment = {}
    for state in range(47):
        c = state % 12
        enviroment[state] = {}
        for action in possible_actions(state):
            enviroment[state][action] = []
            h = actions(state, action)
            for i in range(len(h)):
                reward = (-1 / (10 * (c + 1))) ** 3
                enviroment[state][action].append((1 / 3, reward, next_state(state, h[i])))

    return enviroment


def mdp_algorithm(enviroment, env, gamma=0.9):
    values = [0 for i in range(env.observation_space.n)]
    policy = [possible_actions(i)[0] for i in range(48)]
    t = 1
    while (t < 100):
        q_values, values = update_values(enviroment, env, policy, gamma)
        changed = False

        for state in range(env.observation_space.n):
            mxId = possible_actions(state)[0]
            for action in possible_actions(state):
                if q_values[state][action] > q_values[state][mxId]:
                    mxId = action
            if policy[state] != mxId:
                policy[state] = mxId
                changed = True

        if not changed:
            break

        t += 1
    return values, policy


if __name__ == '__main__':
    main()