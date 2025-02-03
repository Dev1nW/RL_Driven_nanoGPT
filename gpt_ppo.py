import gymnasium as gym
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import ale_py
import os
import torch.nn.functional as F

from model import GPT, GPTConfig

ENV_NAME = "BreakoutNoFrameskip-v4"

NUM_STEPS = 100000          
EVAL_INTERVAL = 100        
EVAL_ITERS = 1         
LOG_INTERVAL = 10         
SAVE_FREQ = 100           

ROLLOUT_STEPS = 2048      
BATCH_SIZE = 64           
EPOCHS = 10               
GAMMA = 0.99              
LAMBD = 0.95             
EPS_CLIP = 0.2           
VALUE_LOSS_COEF = 1.0    

TARGET_KL = 0.04        
LR = 0.0003            
LR_DECAY_ITERS = 2500    
MIN_LR = 0.004           
BETA2 = 0.99             
WARMUP_ITERS = 0         
MAX_GRAD_NORM = 0.5      

BLOCK_SIZE = 8           
N_EMBD = 132            
N_LAYER = 6             
N_HEAD = 6              
DROP_OUT = 0.0
ENTROPY_COEF = 0.0          

EVAL_EPISODES = 10      
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"


def preprocess_observation(obs):
    """Improved preprocessing to retain more information"""
    obs_gray = obs.mean(axis=2)
    obs_gray = obs_gray[::4, ::4]
    obs_gray = ((obs_gray - obs_gray.min()) * 255 / (obs_gray.max() - obs_gray.min() + 1e-8))
    flat_obs = obs_gray.flatten()[:BLOCK_SIZE]
    state_tensor = torch.tensor(flat_obs, dtype=torch.long)
    return state_tensor

class GPTPPOPolicy(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        vocab_size = 256  
        config = GPTConfig(
            block_size=BLOCK_SIZE,
            vocab_size=vocab_size,
            n_layer=N_LAYER, 
            n_head=N_HEAD,
            n_embd=N_EMBD,
            dropout=DROP_OUT,
        )
        self.gpt = GPT(config)
        self.policy_head = nn.Linear(N_EMBD, action_dim)
        self.value_head = nn.Linear(N_EMBD, 1)
        self.post_gpt_linear = nn.Linear(config.vocab_size, N_EMBD)

    def forward(self, obs_sequence):
        logits, _ = self.gpt(obs_sequence)
        feats = logits[:, -1, :]
        feats = self.post_gpt_linear(feats)
        action_logits = self.policy_head(feats)
        value = self.value_head(feats)
        return action_logits, value

class PPOBuffer:
    def __init__(self, size, device='cuda'):
        self.size = size  
        self.cur_idx = 0
        self.obs = []
        self.actions = torch.zeros(size, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(size, device=device)
        self.rewards = torch.zeros(size, device=device)
        self.dones = torch.zeros(size, device=device)
        self.values = torch.zeros(size, device=device)
        self.device = device
        self.advantages = torch.zeros(size, device=device)
        self.returns = torch.zeros(size, device=device)
        
        
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_count = 0

    def store(self, obs, action, logprob, reward, done, value):
        
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std = np.sqrt(self.reward_std**2 + delta * delta2)

        normalized_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        
        self.obs.append(obs)
        self.actions[self.cur_idx] = action
        self.logprobs[self.cur_idx] = logprob
        self.rewards[self.cur_idx] = normalized_reward
        self.dones[self.cur_idx] = done
        self.values[self.cur_idx] = value
        self.cur_idx += 1

    def compute_returns_and_advantages(self, last_value):
        next_values = torch.cat([self.values[1:], torch.tensor([last_value]).to(self.device)])
        deltas = self.rewards + GAMMA * next_values * (1.0 - self.dones) - self.values
        
        gae = 0
        for t in reversed(range(len(self.rewards))):
            gae = deltas[t] + GAMMA * LAMBD * (1.0 - self.dones[t]) * gae
            self.advantages[t] = gae
        
        self.returns = self.advantages + self.values
        
        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)

    def get(self):
        obs_tensor = torch.stack([obs for obs in self.obs], dim=0).to(self.device)
        return (
            obs_tensor,
            self.actions[:self.cur_idx],
            self.logprobs[:self.cur_idx],
            self.advantages[:self.cur_idx],
            self.returns[:self.cur_idx]
        )

def evaluate_policy(env, policy, device, n_episodes=EVAL_EPISODES):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = preprocess_observation(obs).unsqueeze(0).to(device)
                action_logits, _ = policy(state_tensor)
                action = torch.argmax(action_logits, dim=-1)
            
            obs, reward, done, truncated, _ = env.step(action.item())
            episode_reward += reward
            if truncated:
                break
        
        rewards.append(episode_reward)
    
    return np.mean(rewards), np.std(rewards)

def ppo_update(policy, optimizer, obs, actions, old_logprobs, advantages, returns):
    """
    Update policy using the PPO algorithm.
    
    Args:
        policy: The policy model
        optimizer: The optimizer
        obs: Observations
        actions: Actions taken
        old_logprobs: Log probabilities of actions under old policy
        advantages: Computed advantages
        returns: Computed returns
    """
    policy.train()
    
    policy_losses = []
    value_losses = []
    entropy_losses = []
    clip_fractions = []
    approx_kl_divs = []
    
    dataset_size = obs.size(0)
    indices = np.arange(dataset_size)
    
    for epoch in range(EPOCHS):
        np.random.shuffle(indices)
        
        for start in range(0, dataset_size, BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_idx = indices[start:end]
            
            obs_batch = obs[batch_idx]
            actions_batch = actions[batch_idx]
            old_logprob_batch = old_logprobs[batch_idx]
            advantages_batch = advantages[batch_idx]
            returns_batch = returns[batch_idx]
            
            action_logits, values = policy(obs_batch)
            dist = Categorical(logits=action_logits)
            new_logprob = dist.log_prob(actions_batch)
            entropy = dist.entropy().mean()
            
            ratio = (new_logprob - old_logprob_batch).exp()
            policy_loss_1 = advantages_batch * ratio
            policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            
            value_loss = F.mse_loss(values.squeeze(-1), returns_batch)
            
            entropy_loss = -entropy
            
            total_loss = (
                policy_loss + 
                VALUE_LOSS_COEF * value_loss + 
                ENTROPY_COEF * entropy_loss
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            
            clip_fraction = (torch.abs(ratio - 1) > EPS_CLIP).float().mean().item()
            approx_kl = ((ratio - 1) - (new_logprob - old_logprob_batch)).mean().item()
            
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
            clip_fractions.append(clip_fraction)
            approx_kl_divs.append(approx_kl)
    
    return (
        np.mean(policy_losses),
        np.mean(value_losses),
        np.mean(entropy_losses)
    )

def collect_rollouts(env, policy, buffer, n_rollout_steps, device):
    """
    Collect experiences using the current policy and fill the rollout buffer.
    
    Args:
        env: The training environment
        policy: The policy to use for collecting rollouts
        buffer: Buffer to fill with rollouts
        n_rollout_steps: Number of steps to collect
        device: Device to use for tensor operations
    
    Returns:
        bool: True if collection completed successfully
    """
    policy.eval()
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    for step in range(n_rollout_steps):
        with torch.no_grad():
            state_tensor = preprocess_observation(obs).unsqueeze(0).to(device)
            action_logits, value = policy(state_tensor)
            
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            next_obs, reward, done, truncated, info = env.step(action.item())
            episode_reward += reward
            episode_length += 1
            
            buffer.store(
                preprocess_observation(obs),
                action.item(),
                log_prob.item(),
                reward,
                float(done or truncated),
                value.item()
            )
            
            obs = next_obs
            
            if done or truncated:
                print(f"Episode finished. Length: {episode_length}, Reward: {episode_reward}")
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
    
    with torch.no_grad():
        _, last_value = policy(preprocess_observation(obs).unsqueeze(0).to(device))
        last_value = last_value.item()
    
    buffer.compute_returns_and_advantages(last_value)
    
    policy.train()
    
    return True

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    env = gym.make(ENV_NAME)
    eval_env = gym.make(ENV_NAME)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = GPTPPOPolicy(action_dim=env.action_space.n).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LR, betas=(0.9, BETA2))
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=LR_DECAY_ITERS,
        eta_min=MIN_LR
    )
    
    total_steps = 0
    updates = 0
    best_eval_reward = float('-inf')
    
    start_time = time.time()
    
    while total_steps < NUM_STEPS:
        buffer = PPOBuffer(ROLLOUT_STEPS, device=device)
        
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
        
        total_steps += ROLLOUT_STEPS
        
        obs_b, actions_b, old_log_b, adv_b, ret_b = buffer.get()
        policy_loss, value_loss, entropy = ppo_update(
            policy, optimizer, obs_b, actions_b, old_log_b, adv_b, ret_b
        )
        
        scheduler.step()
        
        updates += 1

        print(updates)
        
        if updates % LOG_INTERVAL == 0:
            eval_reward, eval_std = evaluate_policy(eval_env, policy, device, n_episodes=EVAL_ITERS)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Update {updates}, Steps: {total_steps}, LR: {current_lr:.6f}")
            print(f"Eval reward: {eval_reward:.2f} ± {eval_std:.2f}")
            print(f"Losses - Policy: {policy_loss:.4f}, Value: {value_loss:.4f}, Entropy: {entropy:.4f}")
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save({
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'update': updates,
                    'eval_reward': eval_reward,
                }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
        
        if updates % SAVE_FREQ == 0:
            torch.save({
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'update': updates,
            }, os.path.join(CHECKPOINT_DIR, f'checkpoint_{updates}.pt'))
    
    env.close()
    eval_env.close()
    print(f"Training finished. Total steps: {total_steps}, Time: {time.time() - start_time:.2f}s")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")

if __name__ == "__main__":
    main()






