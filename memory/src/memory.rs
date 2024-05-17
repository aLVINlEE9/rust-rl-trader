use tch::{kind::INT64_CPU, Tensor};

pub struct ReplayBuffer {
    states: Tensor,
    next_states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    capacity: usize,
    batch_size: usize,
    len: usize,
    i: usize,
}

impl ReplayBuffer {
    pub fn new(
        states: Tensor,
        next_states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        capacity: usize,
        batch_size: usize,
    ) -> Self {
        ReplayBuffer {
            states,
            next_states,
            actions,
            rewards,
            capacity,
            batch_size,
            len: 0,
            i: 0,
        }
    }

    pub fn push(&mut self, state: &Tensor, next_state: &Tensor, action: &Tensor, reward: &Tensor) {
        let i = (self.i % self.capacity) as i64;

        self.states.get(i as _).copy_(state);
        self.next_states.get(i as _).copy_(next_state);
        self.actions.get(i as _).copy_(action);
        self.rewards.get(i as _).copy_(reward);

        self.i += 1;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    pub fn random_batch(&self) -> (Tensor, Tensor, Tensor, Tensor) {
        let batch_size = self.batch_size.min(self.len);
        let batch_indexes = Tensor::randint(self.len as _, [batch_size as _], INT64_CPU);

        let states = self.states.index_select(0, &batch_indexes);
        let next_states = self.next_states.index_select(0, &batch_indexes);
        let actions = self.actions.index_select(0, &batch_indexes);
        let rewards = self.rewards.index_select(0, &batch_indexes);

        (states, next_states, actions, rewards)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use tch::kind::FLOAT_CPU;

    #[test]
    fn test_new_basic() {
        let num_obs = 4;
        let capacity = 100;

        let init_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_next_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_actions = Tensor::zeros([capacity as _, 1], INT64_CPU);
        let init_rewards = Tensor::zeros([capacity as _, 1], FLOAT_CPU);

        let buffer = ReplayBuffer::new(
            init_states,
            init_next_states,
            init_actions,
            init_rewards,
            capacity,
            10,
        );

        assert_eq!(buffer.len, 0);
        assert_eq!(buffer.i, 0);
        assert_eq!(buffer.capacity, capacity);
        assert_eq!(buffer.batch_size, 10);
        assert_eq!(buffer.states.size(), &[capacity as i64, num_obs as i64][..]);
        assert_eq!(
            buffer.next_states.size(),
            &[capacity as i64, num_obs as i64][..]
        );
        assert_eq!(buffer.actions.size(), &[capacity as i64, 1][..]);
        assert_eq!(buffer.rewards.size(), &[capacity as i64, 1][..]);
    }

    #[test]
    fn test_new_buffer_element() {
        let num_obs = 4;
        let capacity = 100;

        let init_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_next_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_actions = Tensor::zeros([capacity as _, 1], INT64_CPU);
        let init_rewards = Tensor::zeros([capacity as _, 1], FLOAT_CPU);

        let init_state_copy = init_states.shallow_clone();
        let init_next_state_copy = init_next_states.shallow_clone();
        let init_action_copy = init_actions.shallow_clone();
        let init_reward_copy = init_rewards.shallow_clone();

        let buffer = ReplayBuffer::new(
            init_states,
            init_next_states,
            init_actions,
            init_rewards,
            capacity,
            10,
        );
        assert_eq!(buffer.states, init_state_copy);
        assert_eq!(buffer.next_states, init_next_state_copy);
        assert_eq!(buffer.actions, init_action_copy);
        assert_eq!(buffer.rewards, init_reward_copy);
    }

    #[test]
    fn test_push_basic() {
        let num_obs = 4;
        let capacity = 100;

        let init_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_next_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_actions = Tensor::zeros([capacity as _, 1], INT64_CPU);
        let init_rewards = Tensor::zeros([capacity as _, 1], FLOAT_CPU);

        let mut buffer = ReplayBuffer::new(
            init_states,
            init_next_states,
            init_actions,
            init_rewards,
            capacity,
            10,
        );

        // Push 1 element
        let state = Tensor::ones([num_obs], FLOAT_CPU);
        let next_state = Tensor::ones([num_obs], FLOAT_CPU);
        let action = Tensor::ones([1], INT64_CPU);
        let reward = Tensor::ones([1], FLOAT_CPU);

        buffer.push(&state, &next_state, &action, &reward);

        assert_eq!(buffer.len, 1);
        assert_eq!(buffer.i, 1);
    }

    #[test]
    fn test_push_basic2() {
        let num_obs = 4;
        let capacity = 100;

        let init_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_next_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_actions = Tensor::zeros([capacity as _, 1], INT64_CPU);
        let init_rewards = Tensor::zeros([capacity as _, 1], FLOAT_CPU);

        let mut buffer = ReplayBuffer::new(
            init_states,
            init_next_states,
            init_actions,
            init_rewards,
            capacity,
            10,
        );

        // Push 1 element
        let state = Tensor::randn([num_obs], FLOAT_CPU);
        let next_state = Tensor::randn([num_obs], FLOAT_CPU);
        let action = Tensor::randint(10, [1], INT64_CPU);
        let reward = Tensor::randn([1], FLOAT_CPU);

        buffer.push(&state, &next_state, &action, &reward);

        assert_eq!(buffer.len, 1);
        assert_eq!(buffer.i, 1);
    }

    #[test]
    fn test_push_element() {
        let num_obs = 4;
        let capacity = 100;

        let init_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_next_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_actions = Tensor::zeros([capacity as _, 1], INT64_CPU);
        let init_rewards = Tensor::zeros([capacity as _, 1], FLOAT_CPU);

        let mut buffer = ReplayBuffer::new(
            init_states,
            init_next_states,
            init_actions,
            init_rewards,
            capacity,
            10,
        );

        // Push 1 element
        let state = Tensor::ones([num_obs], FLOAT_CPU);
        let next_state = Tensor::ones([num_obs], FLOAT_CPU);
        let action = Tensor::ones([1], INT64_CPU);
        let reward = Tensor::ones([1], FLOAT_CPU);

        buffer.push(&state, &next_state, &action, &reward);

        let i = (buffer.i - 1) as i64;
        assert_eq!(buffer.states.get(i as _), state);
        assert_eq!(buffer.next_states.get(i as _), next_state);
        assert_eq!(buffer.actions.get(i as _), action);
        assert_eq!(buffer.rewards.get(i as _), reward);
    }

    #[test]
    fn test_push_element2() {
        let num_obs = 4;
        let capacity = 100;

        let init_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_next_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_actions = Tensor::zeros([capacity as _, 1], INT64_CPU);
        let init_rewards = Tensor::zeros([capacity as _, 1], FLOAT_CPU);

        let mut buffer = ReplayBuffer::new(
            init_states,
            init_next_states,
            init_actions,
            init_rewards,
            capacity,
            10,
        );

        // Push 1 element
        let state = Tensor::randn([num_obs], FLOAT_CPU);
        let next_state = Tensor::randn([num_obs], FLOAT_CPU);
        let action = Tensor::randint(10, [1], INT64_CPU);
        let reward = Tensor::randn([1], FLOAT_CPU);

        buffer.push(&state, &next_state, &action, &reward);

        let i = (buffer.i - 1) as i64;
        assert_eq!(buffer.states.get(i as _), state);
        assert_eq!(buffer.next_states.get(i as _), next_state);
        assert_eq!(buffer.actions.get(i as _), action);
        assert_eq!(buffer.rewards.get(i as _), reward);
    }

    #[test]
    fn test_push_more_elements() {
        let num_obs = 4;
        let capacity = 100;

        let init_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_next_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_actions = Tensor::zeros([capacity as _, 1], INT64_CPU);
        let init_rewards = Tensor::zeros([capacity as _, 1], FLOAT_CPU);

        let mut buffer = ReplayBuffer::new(
            init_states,
            init_next_states,
            init_actions,
            init_rewards,
            capacity,
            10,
        );

        // push random 10 elements
        for _ in 0..10 {
            let state = Tensor::randn([num_obs], FLOAT_CPU);
            let next_state = Tensor::randn([num_obs], FLOAT_CPU);
            let action = Tensor::randint(10, [1], INT64_CPU);
            let reward = Tensor::randn([1], FLOAT_CPU);

            buffer.push(&state, &next_state, &action, &reward);
        }

        assert_eq!(buffer.len, 10);
        assert_eq!(buffer.i, 10);
        assert_eq!(buffer.states.size(), &[capacity as i64, num_obs][..]);
        assert_eq!(buffer.next_states.size(), &[capacity as i64, num_obs][..]);
        assert_eq!(buffer.actions.size(), &[capacity as i64, 1][..]);
        assert_eq!(buffer.rewards.size(), &[capacity as i64, 1][..]);
    }

    #[test]
    fn test_push_more_elements2() {
        let num_obs = 4;
        let capacity = 100;

        let init_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_next_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_actions = Tensor::zeros([capacity as _, 1], INT64_CPU);
        let init_rewards = Tensor::zeros([capacity as _, 1], FLOAT_CPU);

        let mut buffer = ReplayBuffer::new(
            init_states,
            init_next_states,
            init_actions,
            init_rewards,
            capacity,
            10,
        );

        let mut vec_states = Vec::new();
        let mut vec_next_states = Vec::new();
        let mut vec_actions = Vec::new();
        let mut vec_rewards = Vec::new();
        // push random 10 elements
        for _ in 0..10 {
            let state = Tensor::randn([num_obs], FLOAT_CPU);
            let next_state = Tensor::randn([num_obs], FLOAT_CPU);
            let action = Tensor::randint(10, [1], INT64_CPU);
            let reward = Tensor::randn([1], FLOAT_CPU);

            buffer.push(&state, &next_state, &action, &reward);
            vec_states.push(state);
            vec_next_states.push(next_state);
            vec_actions.push(action);
            vec_rewards.push(reward);
        }

        assert_eq!(buffer.len, 10);
        assert_eq!(buffer.i, 10);
        for i in 0..10 {
            assert_eq!(buffer.states.get(i as _), vec_states[i]);
            assert_eq!(buffer.next_states.get(i as _), vec_next_states[i]);
            assert_eq!(buffer.actions.get(i as _), vec_actions[i]);
            assert_eq!(buffer.rewards.get(i as _), vec_rewards[i]);
        }
    }

    #[test]
    fn test_random_batch_basic() {
        let num_obs = 4;
        let capacity = 100;

        let init_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_next_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_actions = Tensor::zeros([capacity as _, 1], INT64_CPU);
        let init_rewards = Tensor::zeros([capacity as _, 1], FLOAT_CPU);

        let mut buffer = ReplayBuffer::new(
            init_states,
            init_next_states,
            init_actions,
            init_rewards,
            capacity,
            10,
        );

        // push random 10 elements
        for _ in 0..10 {
            let state = Tensor::randn([num_obs], FLOAT_CPU);
            let next_state = Tensor::randn([num_obs], FLOAT_CPU);
            let action = Tensor::randint(10, [1], INT64_CPU);
            let reward = Tensor::randn([1], FLOAT_CPU);

            buffer.push(&state, &next_state, &action, &reward);
        }

        let (states, next_states, actions, rewards) = buffer.random_batch();

        assert_eq!(states.size(), &[10, num_obs][..]);
        assert_eq!(next_states.size(), &[10, num_obs][..]);
        assert_eq!(actions.size(), &[10, 1][..]);
        assert_eq!(rewards.size(), &[10, 1][..]);
    }

    #[test]
    fn test_random_batch_element() {
        let num_obs = 4;
        let capacity = 100;

        let init_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_next_states = Tensor::zeros([capacity as _, num_obs as _], FLOAT_CPU);
        let init_actions = Tensor::zeros([capacity as _, 1], INT64_CPU);
        let init_rewards = Tensor::zeros([capacity as _, 1], FLOAT_CPU);

        let mut buffer = ReplayBuffer::new(
            init_states,
            init_next_states,
            init_actions,
            init_rewards,
            capacity,
            4,
        );

        // push random 50 elements
        for _ in 0..50 {
            let state = Tensor::randn([num_obs], FLOAT_CPU);
            let next_state = Tensor::randn([num_obs], FLOAT_CPU);
            let action = Tensor::randint(10, [1], INT64_CPU);
            let reward = Tensor::randn([1], FLOAT_CPU);

            buffer.push(&state, &next_state, &action, &reward);
        }

        let (states, next_states, actions, rewards) = buffer.random_batch();

        assert_eq!(states.size(), &[4, num_obs][..]);
        assert_eq!(next_states.size(), &[4, num_obs][..]);
        assert_eq!(actions.size(), &[4, 1][..]);
        assert_eq!(rewards.size(), &[4, 1][..]);
    }
}
