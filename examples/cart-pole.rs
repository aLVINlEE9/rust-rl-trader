use plotters::prelude::*;
use pyo3::prelude::*;
use rust_rl_trader::algorithm::dqn::agent::DQNAgent;
use rust_rl_trader::config::model_config::DQNHyperParms;
use tch::Tensor;

fn plot_steps(steps: &[usize], plot_name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(plot_name, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_step = *steps.iter().min().unwrap_or(&0);
    let max_step = *steps.iter().max().unwrap_or(&0);

    let mut chart = ChartBuilder::on(&root)
        .caption(plot_name, ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..steps.len(), min_step as i32..max_step as i32)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            steps.iter().enumerate().map(|(i, &step)| (i, step as i32)),
            &RED,
        ))?
        .label("Steps")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    Ok(())
}

fn main() -> PyResult<()> {
    pyo3::prepare_freethreaded_python();

    Python::with_gil(|py| {
        let mut steps = Vec::new();

        // Create a new Python module
        let gym = py.import_bound("gym")?;
        // Create a new environment
        let env = gym.call_method("make", ("CartPole-v1",), None)?;
        // Get the initial state of the environment
        let (state_array, _) = env
            .call_method("reset", (), None)?
            .extract::<(Vec<f64>, Py<PyAny>)>()?;

        // Get the action space of the environment
        let action_space = env.getattr("action_space")?;

        let input_dim = state_array.len();
        let output_dim = action_space.getattr("n")?.extract::<usize>()?;

        // Create a new DQN agent
        let dqn_hyper_params = DQNHyperParms {
            input_dim,
            output_dim,
            hidden_dims: vec![128, 128],
            learing_rate: 1e-4,
            replay_buffer_capacity: 10000,
            replay_buffer_batch_size: 128,
            epsilon: 0.9,
            epsilon_decay: 1000.0,
            epsilon_min: 0.05,
            epsilon_max: 0.9,
            gamma: 0.99,
            tau: 0.005,
        };

        let mut dqn = DQNAgent::new(dqn_hyper_params, false);

        let num_episodes = 600;
        let mut update_cnt = 0.0;

        for _ in 0..num_episodes {
            let mut episode_steps = 0;
            // Get the initial state of the environment
            let (state_array, _) = env
                .call_method("reset", (), None)?
                .extract::<(Vec<f64>, Py<PyAny>)>()?;
            let mut state_tensor =
                Tensor::from_slice(state_array.as_slice()).to_kind(tch::Kind::Float);

            // Loop through the environment
            loop {
                // Get the action from the DQN agent
                let action = dqn.act(&state_tensor, update_cnt);
                // Take a step in the environment
                let step_result = env.call_method("step", (action,), None)?;
                // Extract the step result
                let (next_state, reward, terminated, truncated, _) =
                    step_result.extract::<(Vec<f64>, f64, bool, bool, Py<PyAny>)>()?;
                let done = terminated || truncated;

                // Convert the next state, action, and reward to tensors
                let next_state_tensor = if terminated {
                    Tensor::full_like(&state_tensor, f64::NAN)
                } else {
                    Tensor::from_slice(&next_state).to_kind(tch::Kind::Float)
                };
                let action_tensor = Tensor::from(action).to_kind(tch::Kind::Int64);
                let reward_tensor = Tensor::from(reward).to_kind(tch::Kind::Float);

                // Train the DQN agent
                dqn.train(
                    &state_tensor,
                    &next_state_tensor,
                    &action_tensor,
                    &reward_tensor,
                );

                // Update the state tensor
                state_tensor = next_state_tensor;
                // let _render_output = env.call_method("render", (), None)?;
                episode_steps += 1;
                update_cnt += 1.0;

                if done {
                    steps.push(episode_steps as usize);
                    break;
                }
            }
            if !dqn.model_config.is_inference {
                dqn.model_config.policy_network.save("policy_network.pth")
            }
        }
        plot_steps(&steps, "steps.png").unwrap();

        Ok(())
    })
}
