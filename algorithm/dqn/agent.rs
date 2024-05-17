use super::utils::*;
use config::model_config::{DQNHyperParms, DQNModelConfig};
use rand::Rng;
use tch::Tensor;

pub struct DQNAgent {
    pub model_config: DQNModelConfig,
    pub hyper_params: DQNHyperParms,
}

impl DQNAgent {
    pub fn new(hyper_params: DQNHyperParms, is_inference: bool) -> Self {
        let model_config = DQNModelConfig::new(&hyper_params, is_inference);
        DQNAgent {
            model_config,
            hyper_params,
        }
    }

    pub fn train(&mut self, state: &Tensor, next_state: &Tensor, action: &Tensor, reward: &Tensor) {
        self.model_config
            .replay_buffer
            .push(state, next_state, action, reward);

        if self.model_config.replay_buffer.len() >= self.hyper_params.replay_buffer_batch_size {
            println!("Training the policy network...");
            // Sample a random batch of transitions
            let samples = self.model_config.replay_buffer.random_batch();
            // Optimize the policy network
            self.model_config.policy_network.opt.zero_grad();
            // Compute the loss
            let loss = compute_loss(&samples, &self.model_config, &self.hyper_params);
            // Compute the gradients
            loss.backward();
            // Update the weights
            self.model_config.policy_network.opt.step();
            // Update the target network
            self.model_config
                .target_network
                .soft_copy_from(&self.model_config.policy_network, &self.hyper_params.tau);
        }
    }

    pub fn act(&mut self, state: &Tensor, steps: f64) -> i64 {
        let mut rng = rand::thread_rng();

        // Update epsilon value based on the decay
        self.hyper_params.epsilon = self.hyper_params.epsilon_min
            + (self.hyper_params.epsilon_max - self.hyper_params.epsilon_min)
                * (-steps / self.hyper_params.epsilon_decay).exp();

        // Function to select action based on policy network
        let select_action_from_policy = || {
            tch::no_grad(|| {
                let q_values = self.model_config.policy_network.forward(state);
                q_values.argmax(0, false).int64_value(&[])
            })
        };

        if self.model_config.is_inference {
            // In inference mode, always use the policy network to select the action
            select_action_from_policy()
        } else {
            // In training mode, use epsilon-greedy strategy
            if self.hyper_params.epsilon > rng.gen_range(0.0..1.0) {
                // Explore: Select a random action
                rng.gen_range(0..self.hyper_params.output_dim as i64)
            } else {
                // Exploit: Select the best action according to the policy network
                select_action_from_policy()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_dqn_hyper_params() -> DQNHyperParms {
        DQNHyperParms {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![8, 16],
            learing_rate: 0.0003,
            replay_buffer_capacity: 10000,
            replay_buffer_batch_size: 32,
            epsilon: 1.0,
            epsilon_decay: 1000.0,
            epsilon_min: 0.01,
            epsilon_max: 1.0,
            gamma: 0.99,
            tau: 0.005,
        }
    }

    #[test]
    fn test_dqn_agent() {
        let hyper_params = create_dqn_hyper_params();
        let mut dqn_agent = DQNAgent::new(hyper_params, false);

        let state = Tensor::randn([4], tch::kind::FLOAT_CPU);
        let next_state = Tensor::randn([4], tch::kind::FLOAT_CPU);
        let action = Tensor::randint(2, [1], tch::kind::INT64_CPU);
        let reward = Tensor::randn([1], tch::kind::FLOAT_CPU);
        dqn_agent.train(&state, &next_state, &action, &reward);

        let steps = 1.0;
        let action = dqn_agent.act(&state, steps);
        assert!((0..2).contains(&action));
    }

    #[test]
    fn test_dqn_agent_train() {
        let hyper_params = create_dqn_hyper_params();
        let mut dqn_agent = DQNAgent::new(hyper_params, false);
        let original_weights: Vec<Tensor> = dqn_agent
            .model_config
            .policy_network
            .get_layers()
            .iter()
            .map(|layer| layer.ws.copy())
            .collect();

        for _ in 0..32 {
            let state = Tensor::randn([4], tch::kind::FLOAT_CPU);
            let next_state = Tensor::randn([4], tch::kind::FLOAT_CPU);
            let action = Tensor::randint(2, [1], tch::kind::INT64_CPU);
            let reward = Tensor::randn([1], tch::kind::FLOAT_CPU);

            dqn_agent.train(&state, &next_state, &action, &reward);
        }

        let new_weights: Vec<Tensor> = dqn_agent
            .model_config
            .policy_network
            .get_layers()
            .iter()
            .map(|layer| layer.ws.copy())
            .collect();

        for (original, new) in original_weights.iter().zip(new_weights.iter()) {
            assert!(!original.allclose(new, 1e-6, 1e-6, false));
        }
    }

    #[test]
    fn test_dqn_agent_updates_target() {
        let hyper_params = create_dqn_hyper_params();
        let mut dqn_agent = DQNAgent::new(hyper_params, false);

        let original_weights: Vec<Tensor> = dqn_agent
            .model_config
            .target_network
            .get_layers()
            .iter()
            .map(|layer| layer.ws.copy())
            .collect();

        for _ in 0..32 {
            let state = Tensor::randn([4], tch::kind::FLOAT_CPU);
            let next_state = Tensor::randn([4], tch::kind::FLOAT_CPU);
            let action = Tensor::randint(2, [1], tch::kind::INT64_CPU);
            let reward = Tensor::randn([1], tch::kind::FLOAT_CPU);

            dqn_agent.train(&state, &next_state, &action, &reward);
        }

        let new_weights: Vec<Tensor> = dqn_agent
            .model_config
            .target_network
            .get_layers()
            .iter()
            .map(|layer| layer.ws.copy())
            .collect();

        for (original, new) in original_weights.iter().zip(new_weights.iter()) {
            assert!(!original.allclose(new, 1e-6, 1e-6, false));
        }
    }

    #[test]
    fn test_compute_loss() {
        let hyper_params = create_dqn_hyper_params();
        let mut dqn_agent = DQNAgent::new(hyper_params, false);

        for _ in 0..32 {
            let state = Tensor::randn([4], tch::kind::FLOAT_CPU);
            let next_state = Tensor::randn([4], tch::kind::FLOAT_CPU);
            let action = Tensor::randint(2, [1], tch::kind::INT64_CPU);
            let reward = Tensor::randn([1], tch::kind::FLOAT_CPU);

            dqn_agent.train(&state, &next_state, &action, &reward);
        }

        let samples = dqn_agent.model_config.replay_buffer.random_batch();
        let loss = compute_loss(&samples, &dqn_agent.model_config, &dqn_agent.hyper_params);

        assert!(loss.double_value(&[]) >= 0.0);
    }

    #[test]
    fn test_act() {
        let hyper_params = create_dqn_hyper_params();
        let mut dqn_agent = DQNAgent::new(hyper_params, false);

        let state = Tensor::randn([4], tch::kind::FLOAT_CPU);
        let steps = 1.0;
        let action = dqn_agent.act(&state, steps);

        assert!((0..2).contains(&action));
    }

    #[test]
    fn test_epsilon_decay() {
        let hyper_params = create_dqn_hyper_params();
        let mut dqn_agent = DQNAgent::new(hyper_params, false);

        let state = Tensor::randn([4], tch::kind::FLOAT_CPU);
        for steps in 0..1000 {
            let action = dqn_agent.act(&state, steps as f64);
            assert!((0..2).contains(&action));
        }

        assert!(dqn_agent.hyper_params.epsilon > dqn_agent.hyper_params.epsilon_min);
        assert!(dqn_agent.hyper_params.epsilon < dqn_agent.hyper_params.epsilon_max);
    }
}
