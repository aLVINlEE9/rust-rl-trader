use tch::nn::{self, Module, ModuleT, OptimizerConfig};
use tch::{Device, Tensor};

pub struct Network {
    var_store: nn::VarStore,
    layers: Vec<nn::Linear>,
    _opt: nn::Optimizer,
    learning_rate: f64,
}

impl Clone for Network {
    fn clone(&self) -> Self {
        let mut new = Self::new(
            self.layers.first().unwrap().ws.size()[1] as usize, // input_dim
            self.layers.last().unwrap().ws.size()[0] as usize,  // output_dim
            self.layers
                .iter()
                .take(self.layers.len() - 1)
                .map(|layer| layer.ws.size()[0] as usize)
                .collect(), // hidden_layers
            self.learning_rate,
        );
        new.var_store.copy(&self.var_store).unwrap();
        new
    }
}

impl Network {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        hidden_layers: Vec<usize>,
        learning_rate: f64,
    ) -> Self {
        let device = Device::cuda_if_available();
        let var_store = nn::VarStore::new(device);
        let _opt = nn::Adam::default()
            .build(&var_store, learning_rate)
            .unwrap();

        let mut layers = Vec::new();
        let mut in_dim = input_dim as i64;

        for &hidden_dim in &hidden_layers {
            layers.push(nn::linear(
                var_store.root() / format!("linear{}", layers.len() + 1),
                in_dim,
                hidden_dim as i64,
                Default::default(),
            ));
            in_dim = hidden_dim as i64;
        }

        layers.push(nn::linear(
            var_store.root() / format!("linear{}", layers.len() + 1),
            in_dim,
            output_dim as i64,
            Default::default(),
        ));

        Self {
            var_store,
            layers,
            _opt,
            learning_rate,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = x.shallow_clone();
        for layer in &self.layers {
            x = layer.forward(&x).relu();
        }
        x
    }

    pub fn forward_t(&self, x: &Tensor) -> Tensor {
        let mut x = x.shallow_clone();
        for layer in &self.layers {
            x = layer.forward_t(&x, false).relu();
        }
        x
    }

    pub fn copy_from(&mut self, source: &Self) {
        tch::no_grad(|| {
            for (layer, source_layer) in self.layers.iter_mut().zip(&source.layers) {
                layer.ws.copy_(&source_layer.ws);
                if let (Some(bs), Some(source_bs)) = (layer.bs.as_mut(), source_layer.bs.as_ref()) {
                    bs.copy_(source_bs);
                }
            }
        });
    }

    pub fn soft_copy_from(&mut self, source: &Self, tau: f64) {
        tch::no_grad(|| {
            for (layer, source_layer) in self.layers.iter_mut().zip(&source.layers) {
                layer.ws = layer.ws.f_mul_scalar(1.0 - tau).unwrap()
                    + source_layer.ws.f_mul_scalar(tau).unwrap();
                if let (Some(bs), Some(source_bs)) = (layer.bs.as_mut(), source_layer.bs.as_ref()) {
                    *bs =
                        bs.f_mul_scalar(1.0 - tau).unwrap() + source_bs.f_mul_scalar(tau).unwrap();
                }
            }
        });
    }

    pub fn save(&self, path: &str) {
        self.var_store.save(path).unwrap();
    }

    pub fn load(&mut self, path: &str) {
        self.var_store.load(path).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::izip;

    #[test]
    fn test_new() {
        let input_dim = 4;
        let output_dim = 2;
        let hidden_layers = vec![8, 16, 32];
        let learning_rate = 0.001;

        let network = Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);

        assert_eq!(network.layers.len(), hidden_layers.clone().len() + 1);
        assert_eq!(network.learning_rate, learning_rate);
    }

    #[test]
    fn test_network_layer() {
        let input_dim = 4;
        let output_dim = 2;
        let hidden_layers = vec![8, 16, 32];
        let learning_rate = 0.001;

        let network = Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);

        assert_eq!(
            network.layers[0].ws.size(),
            [hidden_layers[0] as i64, input_dim as i64]
        );
        assert_eq!(
            network.layers[1].ws.size(),
            [hidden_layers[1] as i64, hidden_layers[0] as i64]
        );
        assert_eq!(
            network.layers[2].ws.size(),
            [hidden_layers[2] as i64, hidden_layers[1] as i64]
        );
        assert_eq!(
            network.layers[3].ws.size(),
            [output_dim as i64, hidden_layers[2] as i64]
        );
    }

    #[test]
    fn test_forward() {
        let input_dim = 4;
        let output_dim = 2;
        let hidden_layers = vec![8, 16, 32];
        let learning_rate = 0.001;

        let network = Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);

        let x = Tensor::randn([1, input_dim as i64], (tch::Kind::Float, Device::Cpu));
        let y = network.forward(&x);

        assert_eq!(y.size(), [1, output_dim as i64]);
    }

    #[test]
    fn test_forward_t() {
        let input_dim = 4;
        let output_dim = 2;
        let hidden_layers = vec![8, 16, 32];
        let learning_rate = 0.001;

        let network = Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);

        let x = Tensor::randn([1, input_dim as i64], (tch::Kind::Float, Device::Cpu));
        let y = network.forward_t(&x);

        assert_eq!(y.size(), [1, output_dim as i64]);
    }

    #[test]
    fn test_copy_from() {
        let input_dim = 4;
        let output_dim = 2;
        let hidden_layers = vec![8, 16, 32];
        let learning_rate = 0.001;

        let mut network1 =
            Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);
        let network2 = Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);

        network1.copy_from(&network2);

        for (layer1, layer2) in network1.layers.iter().zip(network2.layers.iter()) {
            assert_eq!(layer1.ws.size(), layer2.ws.size());
        }
        assert_eq!(network1.learning_rate, network2.learning_rate);
    }

    #[test]
    fn test_copy_from2() {
        let input_dim = 4;
        let output_dim = 2;
        let hidden_layers = vec![8, 16, 32];
        let learning_rate = 0.001;

        let mut network1 =
            Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);
        let network2 = Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);

        network1.copy_from(&network2);

        let x = Tensor::randn([1, input_dim as i64], (tch::Kind::Float, Device::Cpu));
        let y1 = network1.forward_t(&x);
        let y2 = network2.forward_t(&x);

        assert_eq!(y1, y2);
    }

    #[test]
    fn test_copy_from_weight() {
        let input_dim = 4;
        let output_dim = 2;
        let hidden_layers = vec![8, 16, 32];
        let learning_rate = 0.001;

        let mut network1 =
            Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);
        let mut network2 =
            Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);

        for layer in network2.layers.iter_mut() {
            layer.ws = Tensor::randn(layer.ws.size(), (tch::Kind::Float, Device::Cpu));
        }

        network1.copy_from(&network2);

        for (layer1, layer2) in network1.layers.iter().zip(network2.layers.iter()) {
            assert!(layer1.ws.allclose(&layer2.ws, 1e-6, 1e-6, false));
        }
    }

    #[test]
    fn test_copy_from_weight2() {
        let input_dim = 4;
        let output_dim = 2;
        let hidden_layers = vec![8, 16, 32];
        let learning_rate = 0.001;

        let network1 = Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);
        let mut network2 =
            Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);

        for _ in 0..50 {
            let x = Tensor::randn([1, input_dim as i64], (tch::Kind::Float, Device::Cpu));
            let _ = network1.forward_t(&x);
        }

        network2.copy_from(&network1);

        for (layer1, layer2) in network1.layers.iter().zip(network2.layers.iter()) {
            assert!(layer1.ws.allclose(&layer2.ws, 1e-6, 1e-6, false));
        }
    }

    #[test]
    fn test_soft_copy_from_weight() {
        let input_dim = 4;
        let output_dim = 2;
        let hidden_layers = vec![8, 16, 32];
        let learning_rate = 0.001;
        let tau = 0.1;

        let mut network1 =
            Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);
        let mut network2 =
            Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);

        for layer in network2.layers.iter_mut() {
            layer.ws = Tensor::randn(layer.ws.size(), (tch::Kind::Float, Device::Cpu));
        }

        let original_weights: Vec<Tensor> = network1
            .layers
            .iter()
            .map(|layer| layer.ws.shallow_clone())
            .collect();

        network1.soft_copy_from(&network2, tau);

        for (layer1, layer2, original_weight) in izip!(
            network1.layers.iter(),
            network2.layers.iter(),
            original_weights.iter()
        ) {
            let expected_weight =
                original_weight * (1.0 - tau) + layer2.ws.f_mul_scalar(tau).unwrap();
            assert!(layer1.ws.allclose(&expected_weight, 1e-6, 1e-6, false));
        }
    }

    #[test]
    fn test_soft_copy_from_weight2() {
        let input_dim = 4;
        let output_dim = 2;
        let hidden_layers = vec![8, 16, 32];
        let learning_rate = 0.001;
        let tau = 0.1;

        let mut network1 =
            Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);
        let network2 = Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);

        for _ in 0..50 {
            let x = Tensor::randn([1, input_dim as i64], (tch::Kind::Float, Device::Cpu));
            let _ = network2.forward_t(&x);
        }

        let original_weights: Vec<Tensor> = network1
            .layers
            .iter()
            .map(|layer| layer.ws.shallow_clone())
            .collect();

        network1.soft_copy_from(&network2, tau);

        for (layer1, layer2, original_weight) in izip!(
            network1.layers.iter(),
            network2.layers.iter(),
            original_weights.iter()
        ) {
            let expected_weight =
                original_weight * (1.0 - tau) + layer2.ws.f_mul_scalar(tau).unwrap();
            assert!(layer1.ws.allclose(&expected_weight, 1e-6, 1e-6, false));
        }
    }

    #[test]
    fn test_save_load() {
        let input_dim = 4;
        let output_dim = 2;
        let hidden_layers = vec![8, 16, 32];
        let learning_rate = 0.001;

        let mut network1 =
            Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);
        let mut network2 =
            Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);

        for layer in network2.layers.iter_mut() {
            layer.ws = Tensor::randn(layer.ws.size(), (tch::Kind::Float, Device::Cpu));
        }

        network1.copy_from(&network2);

        let path = "test1.pth";
        network1.save(path);
        network2.load(path);

        for (layer1, layer2) in network1.layers.iter().zip(network2.layers.iter()) {
            assert!(layer1.ws.allclose(&layer2.ws, 1e-6, 1e-6, false));
        }

        std::fs::remove_file(path).expect("Failed to remove the file after the test");
    }

    #[test]
    fn test_save_load2() {
        let input_dim = 4;
        let output_dim = 2;
        let hidden_layers = vec![8, 16, 32];
        let learning_rate = 0.001;

        let network1 = Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);
        let mut network2 =
            Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);

        for _ in 0..50 {
            let x = Tensor::randn([1, input_dim as i64], (tch::Kind::Float, Device::Cpu));
            let _ = network1.forward_t(&x);
        }

        let path = "test2.pth";
        network1.save(path);
        network2.load(path);

        for (layer1, layer2) in network1.layers.iter().zip(network2.layers.iter()) {
            assert!(layer1.ws.allclose(&layer2.ws, 1e-6, 1e-6, false));
        }

        std::fs::remove_file(path).expect("Failed to remove the file after the test");
    }

    #[test]
    fn test_clone() {
        let input_dim = 4;
        let output_dim = 2;
        let hidden_layers = vec![8, 16, 32];
        let learning_rate = 0.001;

        let network1 = Network::new(input_dim, output_dim, hidden_layers.clone(), learning_rate);
        let network2 = network1.clone();

        for (layer1, layer2) in network1.layers.iter().zip(network2.layers.iter()) {
            assert!(layer1.ws.allclose(&layer2.ws, 1e-6, 1e-6, false));
        }
        assert_eq!(network1.learning_rate, network2.learning_rate);
    }
}
