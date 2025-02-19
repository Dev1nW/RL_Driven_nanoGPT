# Import necessary libraries and modules
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
import random  
import cv2 
from gymnasium.spaces import Text  


BLOCK_SIZE = 1944  # WIN_HEIGHT * (WIN_WIDTH + 1): 24 * 81 = 1944     

# Global hyperparameters and constants used in training
# Number of timesteps for training
NUM_STEPS = 1000000  

# Evaluate the policy every 20 updates
EVAL_INTERVAL = 20    
# Do 10 evaluation runs and compute the mean and standard deviation  
EVAL_ITERS = 10    
# Save a model checkpoint every 100 policy updates               
SAVE_FREQ = 100                

# Number of steps to collect in each rollout before an update
ROLLOUT_STEPS = 1024           
BATCH_SIZE = 64
# Number of epochs for each PPO update
EPOCHS = 10   

######################
### RL Hyperparams ###
######################

# Discount factor for future rewards (determines how much future rewards are valued)
GAMMA = 0.99      
# GAE (Generalized Advantage Estimation) lambda parameter for bias-variance trade-off             
LAMBD = 0.95     
# Clipping parameter for PPO's surrogate objective to limit policy updates              
EPS_CLIP = 0.2       
# Coefficient for the value loss term in the PPO loss function          
VALUE_LOSS_COEF = 1.0    
# Coefficient for the entropy bonus    
ENTROPY_COEF = 0.0             

# Starting learning rate
LR = 0.0001    
# Number of policy updates to decay learning rate over                 
LR_DECAY_ITERS = 100           
# Minumum learnign rate
MIN_LR = 0.00005       
# Second beta parameter for Adam optimizer (controls the moving average of squared gradients)        
BETA2 = 0.95                   
# Maximum norm for gradient clipping to prevent exploding gradients
MAX_GRAD_NORM = 0.5            

# GPT model configuration hyperparameters used for the policy network
N_EMBD = 96                    
N_LAYER = 6                    
N_HEAD = 6                     
DROP_OUT = 0.0                 
MLP_EXPANSION = 1              
              
CHECKPOINT_DIR = "checkpoints"  

###############################################
# BrickBreaker Environment Implementation
###############################################

class BrickBreakerEnv(gym.Env):
    # Define metadata for the environment (e.g., supported render modes and frame rate)
    metadata = {"render_modes": ["human"], "render_fps": 20}
    
    # Define constants for the game window dimensions and layout
    WIN_WIDTH = 80                    # Width of the game window (number of characters)
    WIN_HEIGHT = 24                   # Height of the game window (number of characters)
    INTERIOR_WIDTH = WIN_WIDTH - 2    # Width of the playable area (excluding borders)
    INTERIOR_HEIGHT = WIN_HEIGHT - 2  # Height of the playable area (excluding borders)
    BRICK_ROWS = 7                    # Number of rows of bricks
    BRICK_COLS = 13                   # Number of columns of bricks
    BRICK_WIDTH = 6                   # Width of each brick in characters
    BRICK_HEIGHT = 1                  # Height of each brick (1 character tall)
    GAP_ABOVE_BRICKS = 4              # Gap (in characters) above the bricks area
    GAP_BETWEEN_BRICKS_AND_PADDLE = 10  # Gap (in characters) between the bricks and the paddle

    def __init__(self, render_mode=None):
        super().__init__()  # Initialize the base gym environment
        self.render_mode = render_mode  # Save the render mode (e.g., "human" for console output)
        
        # Calculate maximum text length for ASCII observation (each line plus newline)
        max_length = self.WIN_HEIGHT * (self.WIN_WIDTH + 1)
        # Define the observation space as a Text space with the calculated maximum length
        self.observation_space = Text(max_length=max_length)
        # Define the action space as a discrete space with 3 possible actions (e.g., left, no-op, right)
        self.action_space = gym.spaces.Discrete(3)
        
        # Set paddle properties: speed and width
        self.paddle_speed = 2  # The paddle moves 2 characters per action
        self.paddle_width = 7  # The paddle spans 7 characters in width
        
        # Set the starting y-position for bricks and the paddle
        self.brick_start_y = 1 + self.GAP_ABOVE_BRICKS  # Starting y-coordinate for bricks, leaving a gap at the top
        self.paddle_y = (self.brick_start_y + self.BRICK_ROWS) + self.GAP_BETWEEN_BRICKS_AND_PADDLE  # y-coordinate for paddle

    def reset(self, seed=None, options=None):
        # Reset the environment to an initial state
        if seed is not None:
            np.random.seed(seed)  # Seed NumPy's random number generator for reproducibility
            random.seed(seed)     # Seed Python's random module for reproducibility
        
        self.score = 0        # Reset the player's score to zero
        self.lives = 5        # Set the number of lives to 5
        # Create a 2D NumPy array for bricks; 1 indicates a brick is present
        self.bricks = np.ones((self.BRICK_ROWS, self.BRICK_COLS), dtype=np.int32)
        
        # Initialize the ball's position: start in the middle of the window horizontally and just above the paddle vertically
        self.ball_x = self.WIN_WIDTH // 2
        self.ball_y = self.paddle_y - 1
        
        # Set the ball's horizontal velocity randomly to either -1 or 1, and its vertical velocity to -1 (moving upward)
        self.ball_dx = random.choice([-1, 1])
        self.ball_dy = -1
        
        # Initialize the paddle's horizontal position centered within the interior space
        self.paddle_x = 1 + (self.INTERIOR_WIDTH - self.paddle_width) // 2
        
        # Return the initial observation (ASCII representation of the game state) and an empty info dictionary
        return self._get_ascii(), {}

    def _get_ascii(self):
        # This helper function converts the current game state into an ASCII string for display
        
        # Create an empty grid filled with spaces, with dimensions WIN_HEIGHT x WIN_WIDTH
        grid = [[" " for _ in range(self.WIN_WIDTH)] for _ in range(self.WIN_HEIGHT)]
        
        # Draw the top and bottom borders of the game window with '#'
        for x in range(self.WIN_WIDTH):
            grid[0][x] = "#"  # Top border
            grid[self.WIN_HEIGHT - 1][x] = "#"  # Bottom border
        
        # Draw the left and right borders of the game window with '#'
        for y in range(self.WIN_HEIGHT):
            grid[y][0] = "#"  # Left border
            grid[y][self.WIN_WIDTH - 1] = "#"  # Right border
        
        # Draw the bricks onto the grid
        for i in range(self.BRICK_ROWS):
            for j in range(self.BRICK_COLS):
                # Only draw the brick if it is still present (value 1)
                if self.bricks[i, j] == 1:
                    # Calculate the x and y starting position for the brick
                    brick_x = 1 + j * self.BRICK_WIDTH
                    brick_y = self.brick_start_y + i
                    # Draw each brick using characters; use '|' for the borders and '_' for the inside
                    for bx in range(self.BRICK_WIDTH):
                        ch = "|" if bx == 0 or bx == self.BRICK_WIDTH - 1 else "_"
                        # Ensure we don't draw beyond the right border
                        if brick_x + bx < self.WIN_WIDTH - 1:
                            grid[brick_y][brick_x + bx] = ch
        
        # Draw the paddle on the grid using '=' characters
        for i in range(self.paddle_width):
            if 0 <= self.paddle_x + i < self.WIN_WIDTH - 1:
                grid[self.paddle_y][self.paddle_x + i] = "="
        
        # Draw the ball on the grid as 'O'
        bx = int(round(self.ball_x))
        by = int(round(self.ball_y))
        if 0 <= by < self.WIN_HEIGHT and 0 <= bx < self.WIN_WIDTH:
            grid[by][bx] = "O"
        
        # Convert the 2D grid into a single string with newlines separating rows and return it
        return "\n".join("".join(row) for row in grid)

    def step(self, action):
        # Process an action and update the game state accordingly

        # If action == 1, move the paddle left by subtracting paddle_speed, ensuring it doesn't go beyond the left border
        if action == 1:
            self.paddle_x = max(1, self.paddle_x - self.paddle_speed)
        # If action == 2, move the paddle right by adding paddle_speed, ensuring it doesn't exceed the right border
        elif action == 2:
            self.paddle_x = min(self.WIN_WIDTH - 1 - self.paddle_width, self.paddle_x + self.paddle_speed)
        
        reward = 0  # Initialize reward for this step
        
        # Update ball position based on its current velocity
        new_ball_x = self.ball_x + self.ball_dx
        new_ball_y = self.ball_y + self.ball_dy
        
        # Check for collision with the left wall; if collided, reverse horizontal direction
        if new_ball_x <= 0:
            new_ball_x = 0
            self.ball_dx = -self.ball_dx
        # Check for collision with the right wall; if collided, reverse horizontal direction
        elif new_ball_x >= self.WIN_WIDTH - 1:
            new_ball_x = self.WIN_WIDTH - 1
            self.ball_dx = -self.ball_dx
        
        # Check for collision with the top wall; if collided, reverse vertical direction
        if new_ball_y <= 0:
            new_ball_y = 0
            self.ball_dy = -self.ball_dy
        # If the ball goes below the bottom wall, deduct a life and reset ball and paddle positions
        elif new_ball_y >= self.WIN_HEIGHT - 1:
            self.lives -= 1  # Lose one life
            # If no lives remain, the game is over; return terminal state with score info
            if self.lives <= 0:
                return self._get_ascii(), 0, True, False, {"score": self.score}
            # Reset ball position to the middle and just above the paddle; randomize horizontal direction
            new_ball_x = self.WIN_WIDTH // 2
            new_ball_y = self.paddle_y - 1
            self.ball_dx = random.choice([-1, 1])
            self.ball_dy = -1
            # Reset paddle position to center
            self.paddle_x = 1 + (self.INTERIOR_WIDTH - self.paddle_width) // 2
        
        # Check if the ball hits the paddle. This is determined by matching the ball's y-position with the paddle's y
        if int(new_ball_y) == self.paddle_y and self.paddle_x <= int(new_ball_x) < self.paddle_x + self.paddle_width:
            # Place the ball just above the paddle and reverse the vertical direction (bounce upward)
            new_ball_y = self.paddle_y - 1
            self.ball_dy = -abs(self.ball_dy)
            # Adjust the horizontal velocity based on where the ball hit relative to the center of the paddle
            hit_offset = (new_ball_x - self.paddle_x) - (self.paddle_width / 2)
            self.ball_dx = 1 if hit_offset >= 0 else -1
        
        # Check if the ball collides with a brick
        brick_row = int(new_ball_y) - self.brick_start_y  # Determine which row of bricks the ball might be in
        if 0 <= brick_row < self.BRICK_ROWS:
            for j in range(self.BRICK_COLS):
                brick_x = 1 + j * self.BRICK_WIDTH  # Calculate x position for this brick
                brick_y = self.brick_start_y + brick_row  # y position for the brick row
                # Check if a brick is present and the ball's position overlaps with the brick
                if (self.bricks[brick_row, j] == 1 and 
                    brick_y == int(new_ball_y) and 
                    brick_x <= int(new_ball_x) < brick_x + self.BRICK_WIDTH):
                    self.bricks[brick_row, j] = 0  # Remove the brick (set to 0)
                    self.ball_dy = -self.ball_dy  # Reverse the ball's vertical direction
                    reward += 10               # Reward for breaking a brick
                    self.score += 10           # Increase the player's score
                    break  # Break out after the collision is handled
        
        # Update the ball's position to the newly calculated values
        self.ball_x = new_ball_x
        self.ball_y = new_ball_y
        
        # Check if all bricks have been broken; if so, mark the game as done
        done = (np.sum(self.bricks) == 0)
        if done:
            reward += 50  # Bonus reward for clearing all bricks
        
        # Return the new observation (ASCII representation), the reward, whether the game is done,
        # a 'truncated' flag (False in this case), and additional info (the current score)
        return self._get_ascii(), reward, done, False, {"score": self.score}

    def render(self, mode="human"):
        # Render the current game state
        ascii_obs = self._get_ascii()  # Get the ASCII representation of the state
        if mode == "human":
            print(ascii_obs)  # Print the ASCII grid to the console
            print(f"Score: {self.score}  Lives: {self.lives}")  # Print current score and lives
        return ascii_obs

    def close(self):
        # Clean up any resources (if necessary) when closing the environment
        pass


def preprocess_observation_text(obs):
    """
    Convert ASCII text observation into a tensor of token ids.
    Each character's ASCII code is used as the token id.
    """
    # Convert each character in the observation string to its ASCII integer code
    token_ids = [ord(c) for c in obs]
    # Return a PyTorch tensor of type long containing the token ids
    return torch.tensor(token_ids, dtype=torch.long)


########################
### GPT-based Policy ###
########################
class GPTPPOPolicy(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        
        vocab_size = 256  # Define vocabulary size based on possible ASCII values (0-255)
        # Create a GPTConfig instance with specified parameters, including the block size (sequence length)
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
        
        # Define a linear layer to map GPT features to the action logits for the policy head
        self.policy_head = nn.Linear(N_EMBD, action_dim)
        # Define a linear layer to map GPT features to a scalar value for state-value estimation
        self.value_head = nn.Linear(N_EMBD, 1)
        # Optionally, include an extra linear layer to map GPT logits back to embedding space before heads
        self.post_gpt_linear = nn.Linear(config.vocab_size, N_EMBD)

    def forward(self, obs_sequence):
        """
        Forward pass through the policy network.
        obs_sequence: (batch_size, sequence_length=BLOCK_SIZE) tensor of token ids.
        Returns:
            action_logits: Logits over actions from the policy head.
            value: Estimated state-value from the value head.
        """
        # Pass the observation sequence through the GPT model; 
        # logits shape: (batch, seq_len, vocab_size)
        logits, _ = self.gpt(obs_sequence)
        # Extract features corresponding to the last token in the sequence (most recent observation)
        feats = logits[:, -1, :]
        # Map the extracted features back to embedding space using the post-GPT linear layer
        feats = self.post_gpt_linear(feats)
        # Pass the features through the policy head to obtain action logits
        action_logits = self.policy_head(feats)
        # Pass the features through the value head to get the scalar value estimate
        value = self.value_head(feats)
        return action_logits, value
    
###########################
### Evaluation Function ###
###########################
def evaluate_policy(env, policy, device, n_episodes=10):
    """
    Evaluate the current policy by running it for a number of episodes.
    Returns the mean and standard deviation of the episode rewards.
    """
    rewards = []  # List to store total rewards for each episode
    for _ in range(n_episodes):
        obs, _ = env.reset()  # Reset environment and get initial observation
        done = False
        episode_reward = 0  # Initialize episode reward counter
        while not done:
            # Disable gradient calculations during evaluation for efficiency
            with torch.no_grad():
                # Preprocess observation and add a batch dimension, then move to the correct device
                state_tensor = preprocess_observation_text(obs).unsqueeze(0).to(device)
                # Pass the state through the policy network to obtain action logits and value (value not used here)
                action_logits, _ = policy(state_tensor)
                # Choose the action with the highest logit (greedy selection)
                action = torch.argmax(action_logits, dim=-1)
            # Take the selected action in the environment
            obs, reward, done, truncated, _ = env.step(action.item())
            episode_reward += reward  
            # If the episode is truncated (early termination), mark it as done
            if truncated:
                done = True
        rewards.append(episode_reward)  # Append total reward for the episode
    # Return the mean and standard deviation of the episode rewards over all evaluation episodes
    return np.mean(rewards), np.std(rewards)


####################################################
### PPO Rollout Buffer for Experience Collection ###
####################################################
class PPOBuffer:
    def __init__(self, size, device='cuda'):
        self.size = size            # Maximum number of transitions to store
        self.cur_idx = 0            # Current index pointer for storing transitions
        self.device = device        

        # Initialize storage for transitions:
        self.obs = []  # List to store preprocessed observation tensors (each of shape [BLOCK_SIZE])
        self.actions = torch.zeros(size, dtype=torch.long, device=device)  # Tensor for storing actions taken
        self.logprobs = torch.zeros(size, device=device)  # Tensor for storing log probabilities of actions
        self.rewards = torch.zeros(size, device=device)  # Tensor for storing rewards received (raw values)
        self.dones = torch.zeros(size, device=device)  # Tensor for storing done flags (1 if episode ended, else 0)
        self.values = torch.zeros(size + 1, device=device)  # Tensor for storing estimated state-values (one extra for bootstrapping)
        self.advantages = torch.zeros(size, device=device)  # Tensor for storing advantage estimates computed via GAE
        self.returns = torch.zeros(size, device=device)  # Tensor for storing computed returns (advantages + values)
        

    def store(self, obs, action, logprob, reward, done, value):
        """
        Store a single transition in the buffer.
        Parameters:
            obs: Preprocessed observation tensor.
            action: Action taken (integer).
            logprob: Log probability of the action.
            reward: Reward received.
            done: Whether the episode ended.
            value: Estimated value of the current state.
        """
        self.obs.append(obs)  # Append the observation tensor to the list
        self.actions[self.cur_idx] = action  # Store the action at the current index
        self.logprobs[self.cur_idx] = logprob  # Store the log probability of the action
        self.rewards[self.cur_idx] = reward  # Store the raw reward
        self.dones[self.cur_idx] = done  # Store whether this transition ended an episode
        self.values[self.cur_idx] = value  # Store the estimated state-value
        self.cur_idx += 1  # Move the index pointer to the next position

    def compute_returns_and_advantages(self, last_value):
        """
        Compute the returns and advantages using Generalized Advantage Estimation (GAE).
        Parameters:
            last_value: The estimated state-value for the state following the last transition.
        """
        length = self.cur_idx  # Number of stored transitions
        self.values[length] = last_value  # Bootstrapping: set the value for the state after the final step
        gae = 0.0  # Initialize the GAE accumulator
        # Iterate backwards over the stored transitions to compute advantages
        for t in reversed(range(length)):
            next_non_terminal = 1.0 - self.dones[t]  # 0 if done, 1 if not done
            # Compute the temporal difference error (delta) for time step t
            delta = self.rewards[t] + GAMMA * self.values[t + 1] * next_non_terminal - self.values[t]
            # Update GAE using the delta and discount factors
            gae = delta + GAMMA * LAMBD * next_non_terminal * gae
            self.advantages[t] = gae  # Store the computed advantage
        # Compute returns by adding the advantages to the state-value estimates
        self.returns[:length] = self.advantages[:length] + self.values[:length]
        # Normalize advantages to have mean 0 and unit variance (improves training stability)
        adv_mean = self.advantages[:length].mean()
        adv_std = self.advantages[:length].std()
        self.advantages[:length] = (self.advantages[:length] - adv_mean) / (adv_std + 1e-8)

    def get(self):
        """
        Retrieve the stored transitions as tensors for training.
        Returns:
            obs_tensor: Tensor of observations with shape [length, BLOCK_SIZE].
            actions: Tensor of actions.
            logprobs: Tensor of log probabilities.
            advantages: Tensor of normalized advantages.
            returns: Tensor of returns (discounted rewards).
            old_values: Tensor of old state-value estimates (for value function clipping).
        """
        length = self.cur_idx  # Number of transitions stored
        # Stack the list of observation tensors into a single tensor and ensure it is on the correct device
        obs_tensor = torch.stack(self.obs[:length], dim=0).to(self.device)
        old_values = self.values[:length].clone().detach()  # Clone old values for stability during updates
        return (
            obs_tensor,         # Observations tensor
            self.actions[:length],  # Actions taken
            self.logprobs[:length],  # Log probabilities of the actions
            self.advantages[:length],  # Normalized advantages
            self.returns[:length],     # Computed returns
            old_values          # Old state-value estimates for clipping in the value loss
        )


################################
### PPO Update Step Function ###
################################
def ppo_update(policy, optimizer, scaler, obs, actions, old_logprobs, advantages, returns, old_values):
    """
    Perform a PPO update on the policy network using the collected rollouts.
    Uses mini-batch stochastic gradient descent over several epochs.
    """
    policy.train()  # Set the policy network to training mode
    policy_losses, value_losses, entropy_losses = [], [], []  # Lists to record losses for logging
    clip_fractions, approx_kl_divs = [], []  # Lists to record clipping fraction and approximate KL divergence

    dataset_size = obs.size(0)  # Total number of transitions in the rollout
    indices = np.arange(dataset_size)  # Create an index array for shuffling the data

    entropy_coef = 0.01  # Set a non-zero entropy coefficient to encourage exploration during training

    # Iterate over the defined number of epochs
    for epoch in range(EPOCHS):
        np.random.shuffle(indices)  # Shuffle indices to randomize mini-batches
        # Process mini-batches
        for start in range(0, dataset_size, BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_idx = indices[start:end]  # Select the current mini-batch indices

            # Slice the batch data from the full dataset
            obs_batch = obs[batch_idx]
            actions_batch = actions[batch_idx]
            old_logprob_batch = old_logprobs[batch_idx]
            advantages_batch = advantages[batch_idx]
            returns_batch = returns[batch_idx]
            old_values_batch = old_values[batch_idx]

            # Use automatic mixed precision for efficient computation on GPUs
            with torch.cuda.amp.autocast():
                # Forward pass: compute new action logits and state-value estimates
                action_logits, values = policy(obs_batch)
                # Create a categorical distribution from the logits
                dist = Categorical(logits=action_logits)
                # Compute new log probabilities for the actions taken
                new_logprob = dist.log_prob(actions_batch)
                # Calculate the entropy of the action distribution (for exploration bonus)
                entropy = dist.entropy().mean()
                # Compute the probability ratio (new vs. old log probabilities) for PPO clipping
                ratio = (new_logprob - old_logprob_batch).exp()
                # Calculate the two surrogate losses
                policy_loss_1 = advantages_batch * ratio
                policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP)
                # Use the minimum of the surrogate losses (PPO objective)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Compute the clipped value function loss
                new_value = values.squeeze(-1)  # Remove extra dimensions
                # Clip the value function updates to be within a range of the old values
                value_pred_clipped = old_values_batch + (new_value - old_values_batch).clamp(-EPS_CLIP, EPS_CLIP)
                # Calculate both unclipped and clipped value losses
                value_loss_unclipped = (new_value - returns_batch).pow(2)
                value_loss_clipped = (value_pred_clipped - returns_batch).pow(2)
                # Use the maximum of the two losses (ensuring a conservative update)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Calculate the entropy loss (negative entropy encourages higher entropy)
                entropy_loss = -entropy
                # Combine the policy loss, value loss, and entropy loss into the total loss
                total_loss = policy_loss + VALUE_LOSS_COEF * value_loss + entropy_coef * entropy_loss

            # Zero out the gradients before performing backpropagation
            optimizer.zero_grad()
            # Scale the loss for mixed precision training and perform backward pass
            scaler.scale(total_loss).backward()
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)
            # Clip gradients to prevent them from exploding
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            # Take an optimization step using the scaled gradients
            scaler.step(optimizer)
            # Update the gradient scaler for the next iteration
            scaler.update()

            # Record the losses for logging purposes
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())
            # Compute an approximate KL divergence for monitoring (using a quadratic approximation)
            approx_kl = 0.5 * ((ratio - 1.0)**2).mean().item()
            approx_kl_divs.append(approx_kl)
            # Compute the fraction of samples where the probability ratio was clipped
            clip_fraction = (torch.abs(ratio - 1) > EPS_CLIP).float().mean().item()
            clip_fractions.append(clip_fraction)

        # Compute the mean KL divergence over the last epoch's mini-batches (for monitoring)
        mean_kl = np.mean(approx_kl_divs[-(dataset_size // BATCH_SIZE):])
        # NOTE: The mean KL could be used to adjust the clipping parameter adaptively
        # NOTE: Early stopping could be implemented 

    # Return the mean losses for logging
    return (
        np.mean(policy_losses),
        np.mean(value_losses),
        np.mean(entropy_losses)
    )


###################################
### Rollout Collection Function ###
###################################
def collect_rollouts(env, policy, buffer, n_rollout_steps, device):
    """
    Collect experiences (rollouts) using the current policy.
    Fills the 'buffer' with transitions from interacting with the environment.
    """
    policy.eval()  # Set the policy to evaluation mode (disable dropout, etc.)
    obs, _ = env.reset()  # Reset the environment to get the initial observation
    episode_reward = 0  # Initialize cumulative reward for the episode
    episode_length = 0  # Initialize step counter for the current episode

    # Loop for a fixed number of rollout steps
    for step in range(n_rollout_steps):
        # Preprocess the ASCII observation into a tensor of token ids
        current_state = preprocess_observation_text(obs)
        # Add a batch dimension and move the tensor to the specified device
        state_tensor = current_state.unsqueeze(0).to(device)

        with torch.no_grad():
            # Pass the state through the policy to get action logits and value estimate
            action_logits, value = policy(state_tensor)
            # Create a distribution from the logits to sample an action (for exploration)
            dist = Categorical(logits=action_logits)
            action = dist.sample()  # Sample an action
            log_prob = dist.log_prob(action)  # Get the log probability of the sampled action

        # Take the sampled action in the environment
        next_obs, reward, done, truncated, info = env.step(action.item())
        
        # Update episode reward and length
        episode_reward += reward
        episode_length += 1

        # Store the transition in the rollout buffer
        buffer.store(
            obs=current_state,         # Preprocessed observation
            action=action.item(),        # Action taken (as an integer)
            logprob=log_prob.item(),     # Log probability of the action
            reward=reward,               # Raw reward received
            done=float(done or truncated),  # Whether the episode ended (done or truncated)
            value=value.item()           # Estimated value for the state
        )

        obs = next_obs  # Update the current observation
        # If the episode has ended or was truncated, reset the environment and print episode stats
        if done or truncated:
            print(f"Episode finished. Length: {episode_length}, Reward: {episode_reward}")
            obs, _ = env.reset()  # Reset environment for a new episode
            episode_reward = 0    # Reset episode reward counter
            episode_length = 0    # Reset episode length counter

    # After collecting the rollout, compute the last state value for bootstrapping advantages
    with torch.no_grad():
        state_tensor = preprocess_observation_text(obs).unsqueeze(0).to(device)
        _, last_value = policy(state_tensor)
        last_value = last_value.item()
    # Compute returns and advantages using the last state value
    buffer.compute_returns_and_advantages(last_value)
    policy.train()  # Set the policy back to training mode
    return True  # Indicate that rollout collection was successful


##############################
### Main Training Function ###
##############################
def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Instantiate the BrickBreaker environment for training and evaluation
    env = BrickBreakerEnv(render_mode="human")
    eval_env = BrickBreakerEnv(render_mode="human")

    # Set the device to CUDA if available; otherwise, use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the GPT-based PPO policy with the appropriate action dimension
    policy = GPTPPOPolicy(action_dim=env.action_space.n).to(device)
    # Create the optimizer (Adam) with the policy's parameters and specified learning rate and beta values
    optimizer = optim.Adam(policy.parameters(), lr=LR, betas=(0.9, BETA2))
    # Create a gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    # Set up a cosine annealing learning rate scheduler for gradual decay of the learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=LR_DECAY_ITERS,  # Total number of iterations for decay
        eta_min=MIN_LR         # Minimum learning rate after decay
    )

    total_steps = 0       # Counter for total training steps taken
    updates = 0           # Counter for number of policy updates performed
    best_eval_reward = float('-inf')  # Initialize best evaluation reward to negative infinity
    start_time = time.time()  # Record the start time of training

    # Main training loop: run until the total number of steps exceeds NUM_STEPS
    while total_steps < NUM_STEPS:
        # Create a new PPO rollout buffer for the current iteration
        buffer = PPOBuffer(ROLLOUT_STEPS, device=device)
        # Collect rollouts by interacting with the environment using the current policy
        continue_training = collect_rollouts(
            env=env,
            policy=policy,
            buffer=buffer,
            n_rollout_steps=ROLLOUT_STEPS,
            device=device
        )
        if not continue_training:
            print("Rollout collection terminated early")
            break

        total_steps += ROLLOUT_STEPS  # Update the total number of steps by the number of rollout steps
        # Retrieve the collected transitions from the buffer
        obs_b, actions_b, old_log_b, adv_b, ret_b, old_values_b = buffer.get()

        # Perform a PPO update using the collected data
        policy_loss, value_loss, entropy = ppo_update(
            policy, optimizer, scaler, 
            obs_b, actions_b, old_log_b, adv_b, ret_b, old_values_b
        )
        scheduler.step()  # Step the learning rate scheduler
        updates += 1      # Increment the update counter
        
        # Print the update number, steps and learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Update {updates}, Steps: {total_steps}, LR: {current_lr:.6f}")

        # Evaluate the policy at regular intervals
        if updates % EVAL_INTERVAL == 0:
            eval_reward, eval_std = evaluate_policy(eval_env, policy, device, n_episodes=EVAL_ITERS)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Update {updates}, Steps: {total_steps}, LR: {current_lr:.6f}")
            print(f"Eval reward: {eval_reward:.2f} ± {eval_std:.2f}")
            print(f"Losses - Policy: {policy_loss:.4f}, Value: {value_loss:.4f}, Entropy: {entropy:.4f}")
            # If the evaluation reward is the best seen so far, save the best model checkpoint
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save({
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'update': updates,
                    'eval_reward': eval_reward,
                }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))

        # Save a checkpoint at regular intervals
        if updates % SAVE_FREQ == 0:
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'update': updates,
            }, os.path.join(CHECKPOINT_DIR, f'checkpoint_{updates}.pt'))

    # Close the environments after training is complete
    env.close()
    eval_env.close()
    print(f"Training finished. Total steps: {total_steps}, Time: {time.time() - start_time:.2f}s")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")

# Standard Python boilerplate to run the main function if this file is executed as a script
if __name__ == "__main__":
    main()
