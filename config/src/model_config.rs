use memory::memory::ReplayBuffer;
use network::networks::Network;
use tch::kind::{FLOAT_CPU, INT64_CPU};
use tch::Tensor;

pub struct DQNHyperParms {
    pub input_dim: usize,
    pub output_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub learing_rate: f64,
    pub replay_buffer_capacity: usize,
    pub replay_buffer_batch_size: usize,
    pub epsilon: f64,
    pub epsilon_decay: f64,
    pub epsilon_min: f64,
    pub epsilon_max: f64,
    pub gamma: f64,
    pub tau: f64,
}

pub struct DQNModelConfig {
    pub policy_network: Network,
    pub target_network: Network,
    pub replay_buffer: ReplayBuffer,
    pub is_inference: bool,
}

impl DQNModelConfig {
    pub fn new(dqn_hyper_params: &DQNHyperParms, is_inference: bool) -> Self {
        let policy_network = Network::new(
            &dqn_hyper_params.input_dim,
            &dqn_hyper_params.output_dim,
            &dqn_hyper_params.hidden_dims,
            dqn_hyper_params.learing_rate,
        );
        let target_network = Network::new(
            &dqn_hyper_params.input_dim,
            &dqn_hyper_params.output_dim,
            &dqn_hyper_params.hidden_dims,
            dqn_hyper_params.learing_rate,
        );
        let replay_buffer = ReplayBuffer::new(
            Tensor::zeros(
                [
                    dqn_hyper_params.replay_buffer_capacity as _,
                    dqn_hyper_params.input_dim as _,
                ],
                FLOAT_CPU,
            ),
            Tensor::zeros(
                [
                    dqn_hyper_params.replay_buffer_capacity as _,
                    dqn_hyper_params.input_dim as _,
                ],
                FLOAT_CPU,
            ),
            Tensor::zeros([dqn_hyper_params.replay_buffer_capacity as _, 1], INT64_CPU),
            Tensor::zeros([dqn_hyper_params.replay_buffer_capacity as _, 1], FLOAT_CPU),
            dqn_hyper_params.replay_buffer_capacity,
            dqn_hyper_params.replay_buffer_batch_size,
        );
        DQNModelConfig {
            policy_network,
            target_network,
            replay_buffer,
            is_inference,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dqn_model_config() {
        let dqn_hyper_params = DQNHyperParms {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![256, 256],
            learing_rate: 0.0003,
            replay_buffer_capacity: 10000,
            replay_buffer_batch_size: 32,
            epsilon: 1.0,
            epsilon_decay: 1000.0,
            epsilon_min: 0.01,
            epsilon_max: 1.0,
            gamma: 0.99,
            tau: 0.005,
        };
        let dqn_model_config = DQNModelConfig::new(&dqn_hyper_params, false);
        assert_eq!(dqn_model_config.policy_network.get_layers().len(), 3);
        assert_eq!(dqn_model_config.target_network.get_layers().len(), 3);
        assert_eq!(dqn_model_config.replay_buffer.len(), 0);
        assert!(!dqn_model_config.is_inference);
    }

    #[test]
    fn test_dqn_hyper_params() {
        let dqn_hyper_params = DQNHyperParms {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![256, 256],
            learing_rate: 0.0003,
            replay_buffer_capacity: 10000,
            replay_buffer_batch_size: 32,
            epsilon: 1.0,
            epsilon_decay: 1000.0,
            epsilon_min: 0.01,
            epsilon_max: 1.0,
            gamma: 0.99,
            tau: 0.005,
        };
        assert_eq!(dqn_hyper_params.input_dim, 4);
        assert_eq!(dqn_hyper_params.output_dim, 2);
        assert_eq!(dqn_hyper_params.hidden_dims, vec![256, 256]);
        assert_eq!(dqn_hyper_params.learing_rate, 0.0003);
        assert_eq!(dqn_hyper_params.replay_buffer_capacity, 10000);
        assert_eq!(dqn_hyper_params.replay_buffer_batch_size, 32);
        assert_eq!(dqn_hyper_params.epsilon, 1.0);
        assert_eq!(dqn_hyper_params.epsilon_decay, 1000.0);
        assert_eq!(dqn_hyper_params.epsilon_min, 0.01);
        assert_eq!(dqn_hyper_params.epsilon_max, 1.0);
        assert_eq!(dqn_hyper_params.gamma, 0.99);
        assert_eq!(dqn_hyper_params.tau, 0.005);
    }

    #[test]
    fn test_model_config_network() {
        let dqn_hyper_params = DQNHyperParms {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![128, 256],
            learing_rate: 0.0003,
            replay_buffer_capacity: 10000,
            replay_buffer_batch_size: 32,
            epsilon: 1.0,
            epsilon_decay: 1000.0,
            epsilon_min: 0.01,
            epsilon_max: 1.0,
            gamma: 0.99,
            tau: 0.005,
        };
        let dqn_model_config = DQNModelConfig::new(&dqn_hyper_params, false);
        assert_eq!(dqn_model_config.policy_network.get_layers().len(), 3);
        assert_eq!(dqn_model_config.target_network.get_layers().len(), 3);
        assert_eq!(
            dqn_model_config.policy_network.get_layers()[0].ws.size(),
            [128, 4]
        );
        assert_eq!(
            dqn_model_config.policy_network.get_layers()[1].ws.size(),
            [256, 128]
        );
        assert_eq!(
            dqn_model_config.policy_network.get_layers()[2].ws.size(),
            [2, 256]
        );
        assert_eq!(
            dqn_model_config.target_network.get_layers()[0].ws.size(),
            [128, 4]
        );
        assert_eq!(
            dqn_model_config.target_network.get_layers()[1].ws.size(),
            [256, 128]
        );
        assert_eq!(
            dqn_model_config.target_network.get_layers()[2].ws.size(),
            [2, 256]
        );
        assert_eq!(dqn_model_config.replay_buffer.len(), 0);
    }

    #[test]
    fn test_model_config_buffer() {
        let dqn_hyper_params = DQNHyperParms {
            input_dim: 4,
            output_dim: 2,
            hidden_dims: vec![128, 256],
            learing_rate: 0.0003,
            replay_buffer_capacity: 10000,
            replay_buffer_batch_size: 32,
            epsilon: 1.0,
            epsilon_decay: 1000.0,
            epsilon_min: 0.01,
            epsilon_max: 1.0,
            gamma: 0.99,
            tau: 0.005,
        };
        let dqn_model_config = DQNModelConfig::new(&dqn_hyper_params, false);
        assert_eq!(dqn_model_config.replay_buffer.len(), 0);
        assert_eq!(dqn_model_config.replay_buffer.get_capacity(), 10000);
        assert_eq!(dqn_model_config.replay_buffer.get_batch_size(), 32);
        assert_eq!(
            dqn_model_config.replay_buffer.get_states().size(),
            [10000, 4]
        );
        assert_eq!(
            dqn_model_config.replay_buffer.get_next_states().size(),
            [10000, 4]
        );
        assert_eq!(
            dqn_model_config.replay_buffer.get_actions().size(),
            [10000, 1]
        );
        assert_eq!(
            dqn_model_config.replay_buffer.get_rewards().size(),
            [10000, 1]
        );
        assert_eq!(dqn_model_config.replay_buffer.len(), 0);
    }
}
