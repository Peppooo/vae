#pragma once
#define _USE_MATH_DEFINES
#define NOMINMAX
#include <iostream>
#include <Eigen/Eigen>
#include <random>
#include <SDL3/SDL.h>
#include <nn.h>
#include "mnist.h"
#include "loader.h"

using namespace std;

mt19937 global_rng;
normal_distribution<float> n_dist(0,1);

class Sampling : public Pass { // first half of input vector rapresents means while the other half is log variance (base e)
public:
	Eigen::VectorXf grad_eps;
	Eigen::VectorXf grad_std;
	Eigen::VectorXf forward(const Eigen::VectorXf& in) override {
		size_t out_samples = in.size() / 2;
		auto mean = in.segment(0,out_samples);
		auto std = (in.segment(out_samples,out_samples).array()*0.5f).exp();
		Eigen::VectorXf eps = Eigen::VectorXf::NullaryExpr(out_samples,[]() { return n_dist(global_rng); });
		return mean.array() + eps.array() * std.array();
	}
	Eigen::VectorXf grad_forward(const Eigen::VectorXf& in) override {
		size_t out_samples = in.size() / 2;
		auto mean = in.segment(0,out_samples);
		grad_std = (in.segment(out_samples,out_samples).array() * 0.5f).exp();
		grad_eps = Eigen::VectorXf::NullaryExpr(out_samples,[]() { return n_dist(global_rng); });
		return mean.array() + grad_eps.array() * grad_std.array();
	}
	Eigen::VectorXf compute_gradients(const Eigen::VectorXf& flow_back) override {
		Eigen::VectorXf ret_flow_back(flow_back.size()*2);
		ret_flow_back << flow_back,flow_back.array()*grad_eps.array()*grad_std.array()*0.5;
		return ret_flow_back;
	}
};

class ResidualBlock : public Pass {
public:
	Sequential stack;
	ResidualBlock(size_t dim,size_t scale = 4):stack({}) {
		stack = Sequential({
				new LayerNorm(dim),
				new Dense(dim, dim * scale),
				new GeLU(),
				new Dropout(0.1f),
				new Dense(dim * scale, dim)
			});
	};
	Eigen::VectorXf forward(const Eigen::VectorXf& in) override {
		return 0.1*stack.forward(in) + in;
	}
	Eigen::VectorXf grad_forward(const Eigen::VectorXf& in) override {
		return 0.1*stack.grad_forward(in) + in;
	}
	Eigen::VectorXf compute_gradients(const Eigen::VectorXf& flow_back) override {
		return 0.1*stack.compute_gradients(flow_back) + flow_back;
	}
	size_t param_size() const override {
		return stack.param_size();
	}
	void param_store(float* dest) const override {
		stack.param_store(dest);
	}
	void param_load(float* src) override {
		stack.param_load(src);
	}
};

Sequential encoder({
		new Dense(3072,512),
		new GeLU(),
		new Dropout(0.1f),

		new ResidualBlock(512),
		new GeLU(),
		new Dropout(0.1f),

		new Dense(512,256),
	});

Sequential decoder({
	new Dense(128,512,0.01f),
	new GeLU(),

	new ResidualBlock(512),
	new GeLU(),

	new Dense(512,3072,0.1f),
	new Sigmoid()
});

void draw_buff_rgb(SDL_Renderer* ren,Eigen::VectorXf& buffer,size_t w) {
	SDL_RenderClear(ren);

	buffer *= 255;

	for(int j = 0; j < buffer.size()/3; j++) {
		//Uint8 c = buffer(j)*255;
		float* c = buffer.data() + (j * 3);
		SDL_SetRenderDrawColor(ren,*(c),*(c + 1),*(c + 2),255);
		SDL_RenderPoint(ren,j % w,j / w);
	}

	buffer /= 255;

	SDL_RenderPresent(ren);
}

void draw_buff(SDL_Renderer* ren,Eigen::VectorXf& buffer,size_t w) {
	SDL_RenderClear(ren);
	for(int j = 0; j < buffer.size(); j++) {
		Uint8 c = buffer(j)*255;
		SDL_SetRenderDrawColor(ren,c,c,c,255);
		SDL_RenderPoint(ren,j % w,j / w);
	}
	SDL_RenderPresent(ren);
}

Sampling sampler; // generates samples from input distributions and allows gradient to flow correctly

Sequential model({&encoder,&sampler,&decoder}); // full model
int main() {
	srand(time(0));

	vector<int> labels;
	auto train_X = load_images_rgb("C:\\Users\\pietr\\Desktop\\datasets\\ffhq32\\",&labels,8192);

	model.load_model("model_ffhq32c.bin");

	cout << "model.param_size() = " << model.param_size() << endl;

	SDL_Init(SDL_INIT_VIDEO);

	SDL_Window* win; SDL_Renderer* ren;
	SDL_Window* win2; SDL_Renderer* ren2;
	SDL_CreateWindowAndRenderer("reconstruction",512,512,0,&win,&ren);
	SDL_CreateWindowAndRenderer("original",512,512,0,&win2,&ren2);
	SDL_SetRenderLogicalPresentation(ren,32,32,SDL_LOGICAL_PRESENTATION_INTEGER_SCALE);
	SDL_SetRenderLogicalPresentation(ren2,32,32,SDL_LOGICAL_PRESENTATION_INTEGER_SCALE);
	SDL_HideWindow(win2);
	SDL_Event e;

	int epochs = 262144;
	int batch_size = 32;
	float lr = 0.001;
	float beta = 1.5; // kl divergence contribution to loss function


	vector<size_t> indecies(train_X.size(),0);
	iota(indecies.begin(),indecies.end(),0ull);
	


	for(int epoch = 1; epoch <= epochs; epoch++) {	

		shuffle(indecies.begin(),indecies.end(),global_rng);

		//beta = min(1,(epochs / 50.0));
		
		float cost = 0;

		for(int _i = 0; _i < train_X.size(); _i += 1) {
			auto i = indecies[_i];

			auto latent_dist = encoder.grad_forward(train_X[i]);
			auto latent = sampler.grad_forward(latent_dist);
			auto reconstruction = decoder.grad_forward(latent);

			Eigen::VectorXf d_cost = (reconstruction - train_X[i]).unaryExpr(&_sign);// l1 cost function, generates sharper images while l2 generates blurrier 

			cost += (reconstruction - train_X[i]).array().abs().mean();

			Eigen::VectorXf decoder_grad = decoder.compute_gradients(d_cost);
			Eigen::VectorXf sampler_grad = sampler.compute_gradients(decoder_grad);

			Eigen::VectorXf kl_div(latent_dist.size());

			size_t h_dim = latent_dist.size() / 2;
			kl_div << latent_dist.segment(0,h_dim),(latent_dist.segment(h_dim,h_dim).array().exp()-1.0f)*0.5f;

			encoder.compute_gradients(sampler_grad + beta * kl_div);


			if(_i % batch_size == 0 && _i != 0) {
				model.apply_gradients(lr);
			}
		}

		model.apply_gradients(lr);

		model.save_model("model_ffhq32c.bin");


		Eigen::VectorXf buffer = decoder.forward(Eigen::VectorXf::NullaryExpr(128,[]() { return n_dist(global_rng); }));
		
		draw_buff_rgb(ren,buffer,32);

		Eigen::VectorXf avg_mu = Eigen::VectorXf::Zero(128),max_var = Eigen::VectorXf::Ones(128) * -INFINITY;
		for(int i = 0; i < train_X.size(); i++) {
			auto lat = encoder.forward(train_X[i]);
			auto mu = lat.segment(0,128),logvar = lat.segment(128,256);
			auto var = (logvar * 0.5).array().exp();
			max_var = max_var.array().max(var);
			avg_mu += mu;
		}
		avg_mu /= train_X.size();

		cost /= train_X.size();
		cout << "epoch " << epoch << " cost " << cost << " latent mu=" << avg_mu.mean() << " var=" << max_var.mean() << endl; // these are averages for mean and max averaged to reduce dimensionality

	}

	return 0;
}
