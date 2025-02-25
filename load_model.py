import gymnasium as gym
from model import GPT, GPTConfig
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import os
import torch.nn.functional as F
import random  # Needed for random.choice and random.seed
import cv2
from gymnasium.spaces import Text
import imageio
from PIL import Image, ImageDraw, ImageFont

# Disable tokenizer parallelism warning.
# Do to fix warning wtih generating video.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BLOCK_SIZE = 1944  # WIN_HEIGHT * (WIN_WIDTH + 1): 24 * 81 = 1944                    
                                    
# For GPT configuration and policy
N_EMBD = 96
N_LAYER = 6
N_HEAD = 6
DROP_OUT = 0.0
MLP_EXPANSION = 1
TARGET_KL = 0.04
     
CHECKPOINT_DIR = "checkpoints"

def preprocess_observation_text(obs):
    """
    Convert ASCII text observation into a tensor of token ids.
    Each character's ASCII code is used as the token id.
    """
    token_ids = [ord(c) for c in obs]
    return torch.tensor(token_ids, dtype=torch.long)

class BrickBreakerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}
    WIN_WIDTH = 80
    WIN_HEIGHT = 24
    INTERIOR_WIDTH = WIN_WIDTH - 2
    INTERIOR_HEIGHT = WIN_HEIGHT - 2
    BRICK_ROWS = 7
    BRICK_COLS = 13
    BRICK_WIDTH = 6
    BRICK_HEIGHT = 1
    GAP_ABOVE_BRICKS = 4
    GAP_BETWEEN_BRICKS_AND_PADDLE = 10

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        max_length = self.WIN_HEIGHT * (self.WIN_WIDTH + 1)
        self.observation_space = Text(max_length=max_length)
        self.action_space = gym.spaces.Discrete(3)
        self.paddle_speed = 2
        self.paddle_width = 7
        self.brick_start_y = 1 + self.GAP_ABOVE_BRICKS
        self.paddle_y = (self.brick_start_y + self.BRICK_ROWS) + self.GAP_BETWEEN_BRICKS_AND_PADDLE

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.score = 0
        self.lives = 5
        self.bricks = np.ones((self.BRICK_ROWS, self.BRICK_COLS), dtype=np.int32)
        self.ball_x = self.WIN_WIDTH // 2
        self.ball_y = self.paddle_y - 1
        self.ball_dx = random.choice([-1, 1])
        self.ball_dy = -1
        self.paddle_x = 1 + (self.INTERIOR_WIDTH - self.paddle_width) // 2
        return self._get_ascii(), {}

    def _get_ascii(self):
        grid = [[" " for _ in range(self.WIN_WIDTH)] for _ in range(self.WIN_HEIGHT)]
        for x in range(self.WIN_WIDTH):
            grid[0][x] = "#"
            grid[self.WIN_HEIGHT - 1][x] = "#"
        for y in range(self.WIN_HEIGHT):
            grid[y][0] = "#"
            grid[y][self.WIN_WIDTH - 1] = "#"
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                if self.bricks[i, j] == 1:
                    brick_x = 1 + j * self.BRICK_WIDTH
                    brick_y = self.brick_start_y + i
                    for bx in range(self.BRICK_WIDTH):
                        ch = "|" if bx == 0 or bx == self.BRICK_WIDTH - 1 else "_"
                        if brick_x + bx < self.WIN_WIDTH - 1:
                            grid[brick_y][brick_x + bx] = ch
        for i in range(self.paddle_width):
            if 0 <= self.paddle_x + i < self.WIN_WIDTH - 1:
                grid[self.paddle_y][self.paddle_x + i] = "="
        bx = int(round(self.ball_x))
        by = int(round(self.ball_y))
        if 0 <= by < self.WIN_HEIGHT and 0 <= bx < self.WIN_WIDTH:
            grid[by][bx] = "O"
        return "\n".join("".join(row) for row in grid)

    def step(self, action):
        if action == 1:
            self.paddle_x = max(1, self.paddle_x - self.paddle_speed)
        elif action == 2:
            self.paddle_x = min(self.WIN_WIDTH - 1 - self.paddle_width, self.paddle_x + self.paddle_speed)
        reward = 0
        new_ball_x = self.ball_x + self.ball_dx
        new_ball_y = self.ball_y + self.ball_dy
        if new_ball_x <= 0:
            new_ball_x = 0
            self.ball_dx = -self.ball_dx
        elif new_ball_x >= self.WIN_WIDTH - 1:
            new_ball_x = self.WIN_WIDTH - 1
            self.ball_dx = -self.ball_dx
        if new_ball_y <= 0:
            new_ball_y = 0
            self.ball_dy = -self.ball_dy
        elif new_ball_y >= self.WIN_HEIGHT - 1:
            self.lives -= 1
            if self.lives <= 0:
                return self._get_ascii(), 0, True, False, {"score": self.score}
            new_ball_x = self.WIN_WIDTH // 2
            new_ball_y = self.paddle_y - 1
            self.ball_dx = random.choice([-1, 1])
            self.ball_dy = -1
            self.paddle_x = 1 + (self.INTERIOR_WIDTH - self.paddle_width) // 2
        if int(new_ball_y) == self.paddle_y and self.paddle_x <= int(new_ball_x) < self.paddle_x + self.paddle_width:
            new_ball_y = self.paddle_y - 1
            self.ball_dy = -abs(self.ball_dy)
            hit_offset = (new_ball_x - self.paddle_x) - (self.paddle_width / 2)
            self.ball_dx = 1 if hit_offset >= 0 else -1
        brick_row = int(new_ball_y) - self.brick_start_y
        if 0 <= brick_row < self.BRICK_ROWS:
            for j in range(self.BRICK_COLS):
                brick_x = 1 + j * self.BRICK_WIDTH
                brick_y = self.brick_start_y + brick_row
                if (self.bricks[brick_row, j] == 1 and 
                    brick_y == int(new_ball_y) and 
                    brick_x <= int(new_ball_x) < brick_x + self.BRICK_WIDTH):
                    self.bricks[brick_row, j] = 0
                    self.ball_dy = -self.ball_dy
                    reward += 10
                    self.score += 10
                    break
        self.ball_x = new_ball_x
        self.ball_y = new_ball_y
        done = (np.sum(self.bricks) == 0)
        if done:
            reward += 50
        return self._get_ascii(), reward, done, False, {"score": self.score}

    def render(self, mode="human"):
        ascii_obs = self._get_ascii()
        if mode == "human":
            print(ascii_obs)
        return ascii_obs

    def close(self):
        pass

class GPTPPOPolicy(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        vocab_size = 256  # for ASCII values (0-255)
        config = GPTConfig(
            block_size=BLOCK_SIZE,  
            vocab_size=vocab_size,
            n_layer=N_LAYER,
            n_head=N_HEAD,
            n_embd=N_EMBD,
            dropout=DROP_OUT,
            mlp_expansion=MLP_EXPANSION
        )
        self.gpt = GPT(config)
        self.policy_head = nn.Linear(N_EMBD, action_dim)
        self.value_head = nn.Linear(N_EMBD, 1)
        # Optionally map GPT logits back to embeddings before heads:
        self.post_gpt_linear = nn.Linear(config.vocab_size, N_EMBD)

    def forward(self, obs_sequence):
        """
        obs_sequence: (batch_size, sequence_length=BLOCK_SIZE) of tokens (long).
        """
        logits, _ = self.gpt(obs_sequence)       # (batch, seq_len, vocab_size)
        feats = logits[:, -1, :]                 # take the last token's logits
        feats = self.post_gpt_linear(feats)      
        action_logits = self.policy_head(feats)  
        value = self.value_head(feats)           
        return action_logits, value


def ascii_to_image(ascii_text: str, num_cols=80, num_rows=24) -> Image.Image:
    """
    Converts an ASCII art string into a properly formatted image using a default monospaced font.
    """
    font = ImageFont.load_default()
    bbox = font.getbbox("A")
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]
    img_width, img_height = num_cols * char_width, num_rows * char_height

    image = Image.new("RGB", (img_width, img_height), "black")
    draw = ImageDraw.Draw(image)
    
    lines = ascii_text.splitlines()
    for row, line in enumerate(lines[:num_rows]):
        draw.text((0, row * char_height), line.ljust(num_cols), font=font, fill="white")
    
    return image

def run_best_model():
    """
    Load the best model from disk and run one episode until termination.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the environment 
    env = BrickBreakerEnv(render_mode="human")
    
    # Instantiate the policy model with the same hyperparameters
    policy = GPTPPOPolicy(action_dim=env.action_space.n).to(device)
    
    # Load the best model checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found at", checkpoint_path)
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    # Run one thousand steps
    obs, _ = env.reset()
    done = False

    total_reward = 0
    frames = []

    for i in range(1000):

        frame_img = ascii_to_image(obs, num_cols=80, num_rows=24)
        frames.append(frame_img)

        # Preprocess the observation (ASCII text to token ids)
        state_tensor = preprocess_observation_text(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            # For evaluation, we use argmax to select the most probable action
            action_logits, _ = policy(state_tensor)
            action = torch.argmax(action_logits, dim=-1).item()

        # Take a step in the environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the environment to display the ASCII observation
        env.render()
        
        if done:
            obs, _ = env.reset(seed=42)
            frame_img = ascii_to_image(obs, num_cols=80, num_rows=24)
            frames.append(frame_img)
            done = False
    
    print("Episode finished, total reward:", total_reward)

    print("Saving video of performance...")
    video_filename = "breakout_1.mp4"
    images = [np.array(frame) for frame in frames if frame is not None]
    imageio.mimwrite(video_filename, images, fps=env.metadata["render_fps"])
    print(f"Saved video as {video_filename}")

    env.close()

if __name__ =="__main__":
    run_best_model()