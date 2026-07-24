#define _USE_MATH_DEFINES
#include <iostream>
#include <Eigen/Eigen>
#include <random>
#include <SDL3/SDL.h>
#include "nn.h"
#include "mnist.h"

using namespace std;

mt19937 global_rng;
normal_distribution<double> n_dist(0,1);

class Sampling : public Pass { // first half of input vector rapresents means while the other half is log variance (base e)
public:
	Eigen::VectorXd grad_eps;
	Eigen::VectorXd grad_std;
	Eigen::VectorXd forward(const Eigen::VectorXd& in) override {
		size_t out_samples = in.size() / 2;
		auto mean = in.segment(0,out_samples);
		auto std = (in.segment(out_samples,out_samples).array()*0.5).exp();
		Eigen::VectorXd eps = Eigen::VectorXd::NullaryExpr(out_samples,[]() { return n_dist(global_rng); });
		return mean.array() + eps.array() * std.array();
	}
	Eigen::VectorXd grad_forward(const Eigen::VectorXd& in) override {
		size_t out_samples = in.size() / 2;
		auto mean = in.segment(0,out_samples);
		grad_std = (in.segment(out_samples,out_samples).array() * 0.5).exp();
		grad_eps = Eigen::VectorXd::NullaryExpr(out_samples,[]() { return n_dist(global_rng); });
		return mean.array() + grad_eps.array() * grad_std.array();
	}
	void reset_gradients() override {};
	void apply_gradients(double lr) override {};
	Eigen::VectorXd compute_gradients(const Eigen::VectorXd& flow_back) override {
		Eigen::VectorXd ret_flow_back(flow_back.size()*2);
		ret_flow_back << flow_back,flow_back.array()*grad_eps.array()*grad_std.array()*0.5;
		return ret_flow_back;
	}
};

int main() {
	Sequential encoder = {{
		new Dense(784,256),
		new ReLU(),
		new Dropout(0.4),

		new Dense(256,128),
		new ReLU(),
		new Dropout(0.3),

		new Dense(128,32),
	}};

	Sequential decoder = {{
		new Dense(16,128),
		new LeakyReLU(0.05),
		new Dropout(0.3),

		new Dense(128,256),
		new LeakyReLU(0.05),
		new Dropout(0.4),

		new Dense(256,784),
		new Sigmoid()
	}};

	Sampling sampler; // converts the 32 values to 16 samples from input distributions and allows gradient to flow correctly

	vector<Eigen::Vector<double,784>> train_X,test_X;


	mnist::read_images("C:\\Users\\pietr\\source\\repos\\neural-net\\training\\digits\\train-images.idx3-ubyte",train_X);

	mnist::read_images("C:\\Users\\pietr\\source\\repos\\neural-net\\training\\digits\\test-images.idx3-ubyte",test_X);

	int epochs = 128;
	int batch_size = 64;
	double lr = 0.03;
	double beta = 0.01; // kl divergence contribution to loss function

	vector<size_t> indecies(train_X.size(),0);
	iota(indecies.begin(),indecies.end(),0ull);

	for(int epoch = 1; epoch <= epochs; epoch++) {
		double cost = 0;

		shuffle(indecies.begin(),indecies.end(),global_rng);
		for(int _i = 0; _i < train_X.size(); _i += 1) {
			auto i = indecies[_i];
			
			auto latent_dist = encoder.grad_forward(train_X[i]);
			auto latent = sampler.grad_forward(latent_dist);
			auto reconstruction = decoder.grad_forward(latent);

			// derivative of mse in respect to p is 2*(p-y)

			Eigen::VectorXd d_cost = 2*(reconstruction - train_X[i]);
			
			cost += (reconstruction - train_X[i]).array().pow(2).mean();

			Eigen::VectorXd decoder_grad = 
			decoder.compute_gradients(d_cost);
			Eigen::VectorXd sampler_grad = sampler.compute_gradients(decoder_grad);
			
			Eigen::VectorXd kl_div(latent_dist.size());

			size_t h_dim = latent_dist.size() / 2;
			kl_div << 2*latent_dist.segment(0,h_dim),(latent_dist.segment(h_dim,h_dim).array().exp()-1);

			encoder.compute_gradients(sampler_grad+beta*kl_div);

			if(_i % batch_size == 0 && _i != 0) {
				encoder.apply_gradients(lr);
				decoder.apply_gradients(lr);
			}
		}
		
		encoder.apply_gradients(lr); // apply possible remaining gradients
		decoder.apply_gradients(lr);


		/*double weights_sq_sum = ((Dense*)model.layers[0])->weights.array().pow(2).sum() +
			((Dense*)model.layers[3])->weights.array().pow(2).sum() +
			((Dense*)model.layers[6])->weights.array().pow(2).sum();*/
		cost /= train_X.size();
		cout << "epoch " << epoch << " cost " << cost << endl;
	}
}