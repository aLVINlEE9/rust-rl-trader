use crate::config::model_config::{DQNHyperParms, DQNModelConfig};
use crate::network::networks::Network;
use tch::{Reduction, Tensor};

pub fn compute_loss(
    samples: &(Tensor, Tensor, Tensor, Tensor),
    model_config: &DQNModelConfig,
    hyper_params: &DQNHyperParms,
) -> Tensor {
    let curr_q_values = model_config
        .policy_network
        .forward_t(&samples.0)
        .gather(1, &samples.2, false);

    let next_q_value = tch::no_grad(|| {
        let next_q_values = model_config.target_network.forward_t(&samples.1);
        let (max_values, _) = next_q_values.max_dim(1, false);
        max_values.unsqueeze(1)
    });
    let next_q_value = next_q_value.nan_to_num(0.0, f64::INFINITY, f64::NEG_INFINITY);

    let target = next_q_value * hyper_params.gamma + &samples.3;
    curr_q_values.smooth_l1_loss(&target, Reduction::Mean, 1.0)
}

pub fn clip_grad_value(network: &mut Network, clip_value: f64) {
    for (_, param) in network.var_store.variables() {
        let _ = param.grad().clamp_(-clip_value, clip_value);
    }
}
