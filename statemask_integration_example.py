"""
Complete StateMask Integration Example

This script demonstrates how to integrate StateMask with the STORM architecture
for decoupled learning of action blinding strategies.
"""

import torch
import yaml
from statemask_trainer import create_statemask_trainer, StateMaskTrainer
from sub_models.statemask import StateMaskGate
import agents
from train import build_world_model, build_agent


def load_config(config_path: str):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class STORMWithStateMask:
    """STORM training with integrated StateMask for action blinding."""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        
        # Build core components
        self.world_model = build_world_model(self.config)
        self.agent = build_agent(self.config, action_dim=6)  # Pong has 6 actions
        
        # Calculate feature dimension for StateMask
        self.feat_dim = 32*32 + self.config['Models']['WorldModel']['TransformerHiddenDim']
        
        # Create StateMask and trainer
        statemask_config = {
            'hidden_dim': 128,
            'lr': 1e-4,
            'fidelity_weight': 1.0,
            'sparsity_weight': 0.1,
            'target_sparsity': 0.3,
            'train_frequency': 100,
            'batch_size': 256
        }
        
        self.statemask, self.statemask_trainer = create_statemask_trainer(
            feat_dim=self.feat_dim,
            config=statemask_config
        )
        
        print(f"✅ Initialized STORM with StateMask")
        print(f"   Feature dimension: {self.feat_dim}")
        print(f"   Target sparsity: {statemask_config['target_sparsity']}")
        
    def train_step_with_statemask(self, obs_batch, action_batch, reward_batch, 
                                termination_batch, use_statemask=True, logger=None):
        """Training step that includes StateMask experience collection and training."""
        
        # 1. Regular world model and agent training
        world_model_metrics = self.world_model.update(
            obs_batch, action_batch, reward_batch, termination_batch, logger=logger
        )
        
        # 2. Imagination with StateMask (if enabled)
        statemask_for_imagination = self.statemask if use_statemask else None
        
        # Sample recent experience for imagination
        sample_obs = obs_batch[:, -8:]  # Last 8 timesteps as context
        sample_action = action_batch[:, -8:]
        
        imagine_latent, imagine_action, imagine_reward, imagine_termination = self.world_model.imagine_data(
            agent=self.agent,
            sample_obs=sample_obs,
            sample_action=sample_action,
            imagine_batch_size=1024,
            imagine_batch_length=16,
            log_video=False,
            logger=logger,
            statemask=statemask_for_imagination
        )
        
        # 3. Agent training on imagined data
        # Note: You'd need to collect old_logprob and old_value for PPO-style training
        # This is simplified for demonstration
        
        # 4. Collect experience for StateMask training
        if use_statemask:
            # Get current policy logits for fidelity calculation
            with torch.no_grad():
                policy_logits = self.agent.policy(imagine_latent[:, :-1])
            
            # Collect experience for decoupled StateMask training
            self.statemask_trainer.collect_experience(
                states=imagine_latent[:, :-1],  # States
                logits=policy_logits,           # Policy logits
                actions=imagine_action          # Actions taken
            )
            
            # 5. Train StateMask (decoupled from main RL objective)
            statemask_metrics = self.statemask_trainer.train_step()
            
            if logger and statemask_metrics:
                for key, value in statemask_metrics.items():
                    logger.log(key, value)
        
        return world_model_metrics
    
    def evaluate_statemask_performance(self, eval_obs, eval_actions, logger=None):
        """Evaluate StateMask performance on evaluation data."""
        
        # Encode evaluation observations
        with torch.no_grad():
            eval_latent = self.world_model.encode_obs(eval_obs)
            context_length = eval_obs.shape[1]
            
            # Process through transformer to get hidden states
            self.world_model.storm_transformer.reset_kv_cache_list(
                eval_obs.shape[0], dtype=torch.float32
            )
            
            for i in range(context_length):
                _, _, _, _, last_dist_feat = self.world_model.predict_next(
                    eval_latent[:, i:i+1],
                    eval_actions[:, i:i+1] if i < eval_actions.shape[1] else torch.zeros_like(eval_actions[:, 0:1]),
                    log_video=False
                )
            
            # Combine latent and hidden features
            eval_states = torch.cat([eval_latent[:, -1:], last_dist_feat], dim=-1)
            eval_states = eval_states.squeeze(1)  # Remove time dimension
            
            # Get policy logits
            eval_logits = self.agent.policy(eval_states)
            
            # Evaluate StateMask
            eval_metrics = self.statemask_trainer.evaluate_masking_quality(
                eval_states, eval_logits
            )
            
            # Get blinding statistics
            blinding_stats = self.statemask_trainer.get_blinding_statistics(eval_states)
            
            # Combine metrics
            all_metrics = {**eval_metrics, **blinding_stats}
            
            if logger:
                for key, value in all_metrics.items():
                    logger.log(key, value)
            
            return all_metrics
    
    def adaptive_sparsity_schedule(self, training_step: int, total_steps: int):
        """Implement adaptive sparsity scheduling during training."""
        
        # Example: Start with low sparsity, gradually increase
        progress = training_step / total_steps
        
        if progress < 0.3:
            # Initial phase: Low sparsity (0.1)
            target_sparsity = 0.1
        elif progress < 0.7:
            # Middle phase: Gradual increase (0.1 -> 0.4)
            target_sparsity = 0.1 + 0.3 * ((progress - 0.3) / 0.4)
        else:
            # Final phase: High sparsity (0.4)
            target_sparsity = 0.4
            
        self.statemask_trainer.set_target_sparsity(target_sparsity)
        
        return target_sparsity
    
    def save_statemask(self, path: str):
        """Save trained StateMask."""
        torch.save({
            'statemask_state_dict': self.statemask.state_dict(),
            'trainer_config': {
                'feat_dim': self.feat_dim,
                'target_sparsity': self.statemask_trainer.target_sparsity,
                'fidelity_weight': self.statemask_trainer.fidelity_weight,
                'sparsity_weight': self.statemask_trainer.sparsity_weight,
            }
        }, path)
        print(f"✅ Saved StateMask to {path}")
    
    def load_statemask(self, path: str):
        """Load pre-trained StateMask."""
        checkpoint = torch.load(path)
        self.statemask.load_state_dict(checkpoint['statemask_state_dict'])
        print(f"✅ Loaded StateMask from {path}")


def demo_statemask_training():
    """Demonstration of StateMask training integration."""
    
    print("🚀 Starting StateMask Integration Demo")
    
    # Initialize STORM with StateMask
    storm = STORMWithStateMask('config_files/STORM.yaml')
    
    # Simulate training loop
    total_steps = 1000
    
    for step in range(total_steps):
        # Simulate batch data (normally from replay buffer)
        batch_size, seq_len = 16, 64
        obs_batch = torch.randn(batch_size, seq_len, 3, 64, 64).cuda()
        action_batch = torch.randint(0, 6, (batch_size, seq_len)).cuda()
        reward_batch = torch.randn(batch_size, seq_len).cuda()
        termination_batch = torch.zeros(batch_size, seq_len).cuda()
        
        # Training step with StateMask
        use_statemask = step > 100  # Start using StateMask after 100 steps
        
        # Adaptive sparsity scheduling
        if use_statemask:
            target_sparsity = storm.adaptive_sparsity_schedule(step, total_steps)
            if step % 100 == 0:
                print(f"Step {step}: Target sparsity = {target_sparsity:.3f}")
        
        # Perform training step
        metrics = storm.train_step_with_statemask(
            obs_batch, action_batch, reward_batch, termination_batch, 
            use_statemask=use_statemask
        )
        
        # Evaluation every 200 steps
        if step % 200 == 0 and step > 0:
            eval_metrics = storm.evaluate_statemask_performance(
                obs_batch[:4, -16:], action_batch[:4, -16:]  # Small eval batch
            )
            
            if use_statemask:
                print(f"📊 Step {step} Evaluation:")
                print(f"   Blinding rate: {eval_metrics.get('blinding_rate', 0):.3f}")
                print(f"   Fidelity loss: {eval_metrics.get('eval_statemask/fidelity_loss', 0):.4f}")
    
    # Save trained StateMask
    storm.save_statemask('trained_statemask.pth')
    
    print("✅ StateMask training demo completed!")


if __name__ == "__main__":
    demo_statemask_training()
