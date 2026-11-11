#define NOMINMAX
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"

#include <iostream>
#include <random>
#include <algorithm>
#include <SDL3/SDL.h>
#include <chrono>

#include "C:\Users\pietr\source\repos\cu-nn\cu-nn\nn.cuh"
#include "C:\Users\pietr\source\repos\cu-nn\cu-nn\optim.cuh"
#include "C:\Users\pietr\source\repos\cu-nn\cu-nn\timer.h"
#include <nn/loader.h>
#include <nn/mnist.h>

#include <nlohmann/json.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using namespace std;

mt19937 global_rng;
normal_distribution<float> n_dist(0,1);

constexpr size_t image_w = 32;
constexpr size_t image_dim = image_w * image_w * 3;
constexpr size_t latent_dim = 128;

//Sequential encoder({ // small model
//	new Dense(image_dim,512),
//	new GeLU(),
//	new Dropout(0.3f), 
//
//	new ResidualBlock(512,new GeLU,2,0.3f,0.1f),
//	new GeLU(),
//	new Dropout(0.4f),
//
//	new ResidualBlock(512,new GeLU,2,0.3f,0.1f),
//	new GeLU(),
//	new Dropout(0.3f),
//
//	new Dense(512,latent_dim * 2),
//	});
//
//Sequential decoder({
//	new Dense(latent_dim,512),
//	new GeLU(),
//
//	new ResidualBlock(512,new GeLU,2,0.0f,0.1f),
//	new GeLU(),
//
//	new ResidualBlock(512,new GeLU,2,0.0f,0.1f),
//	new GeLU(),
//
//	new Dense(512,image_dim),
//	new Sigmoid()
//});

Sequential encoder({ // large model
	new Dense(image_dim,1024),
	new ReLU(),
	new Dropout(0.1f),

	new ResidualBlock(1024,new GeLU,2,0.1f,0.1f),
	new GeLU(),
	new Dropout(0.1f),

	new ResidualBlock(1024,new GeLU,2,0.1f,0.05f),
	new GeLU(),
	new Dropout(0.1f),

	new ResidualBlock(1024,new GeLU,2,0.1f,0.05f),
	new GeLU(),
	new Dropout(0.1f),

	new Dense(1024,latent_dim * 2),
});
Sequential decoder({
	new Dense(latent_dim,1024),
	new GeLU(),

	new ResidualBlock(1024,new GeLU,2,0.0f,0.1f),
	new GeLU(),

	new ResidualBlock(1024,new GeLU,2,0.0f,0.05f),
	new GeLU(),

	new ResidualBlock(1024,new GeLU,2,0.0f,0.05f),
	new GeLU(),

	new Dense(1024,image_dim),
	new Sigmoid()
});

Sampling sampler(latent_dim);

int main() {
	init_cublas();
	SDL_Init(SDL_INIT_VIDEO);
	nn_rand::global_rng.seed(time(0));
	global_rng.seed(time(0) * 2);

	SDL_Window* win;
	SDL_Renderer* ren;
	SDL_CreateWindowAndRenderer("sample",512,512,0,&win,&ren);

	Sequential model({&encoder,&sampler,&decoder});

	auto latent_noise = [](int cols = 1) {
		Tensor vec = Eigen::MatrixXf::NullaryExpr(latent_dim,cols,[]() { return n_dist(global_rng); });
		vec.sync_gpu();
		return vec;
	};

	cout << "model.param_size() = " << model.param_size() << endl;

	model.load_model("resdense-ffhq32-large.bin");

	for(;;) {
		auto img = decoder.forward(latent_noise());
		img.sync_cpu();
		Eigen::VectorXf buff = img.data.col(0);
		draw_buff_rgb(ren,buff,image_w,UpscaleMode::NEAREST);
		SDL_RenderPresent(ren);
		SDL_Delay(500);
	}

	//vector<Eigen::Vector<float,image_dim>> raw_train_X;
	auto raw_train_X = load_images_rgb("C:\\Users\\pietr\\Desktop\\datasets\\ffhq32",nullptr);
	int epochs = 200;

	const int batches = 2000;//9735;
	const int batch_size = raw_train_X.size() / batches;

	float lr = 3e-5f;
	float beta = 0.7f;

	AdamW optim(&model,lr,0.001f,0.6f);
	GradClip grad_clip(&model,0.1f);

	vector<Tensor> train_X(batches);
	for(int i = 0; i < batches; i++) {
		train_X[i] = Tensor(image_dim,batch_size,true);
	}

	timer _timer;

	vector<int> indecies(batches);
	iota(indecies.begin(),indecies.end(),0);

	for(int epoch = 0; epoch < epochs; epoch++) {
		Tensor fixed_noise = latent_noise(4);

		beta = max(min(epoch / 10.0f,1.5f),0.8f);

		shuffle(raw_train_X.begin(),raw_train_X.end(),global_rng);

		for(int i = 0; i < batches; i++) {
			for(int j = 0; j < batch_size; j++) {
				cudaMemcpyAsync(train_X[i].gpu() + j * image_dim,raw_train_X[i * batch_size + j].data(),image_dim * sizeof(float),cudaMemcpyHostToDevice);
			}
			train_X[i].sync_cpu();
		}
		cudaDeviceSynchronize();

		Eigen::VectorXf latent_info = Eigen::VectorXf::Zero(latent_dim * 2);

		float cost = 0;

		shuffle(indecies.begin(),indecies.end(),global_rng);

		for(int i = 0; i < batches; i++) {
			float batch_time = _timer.end();
			_timer.start();

			int batch = indecies[i];

			auto latent_dist = encoder.grad_forward(train_X[batch]);

			latent_dist.sync_cpu();

			latent_info.segment(0,latent_dim) += latent_dist.data.rowwise().sum().segment(0,latent_dim);
			latent_info.segment(latent_dim,latent_dim) += (latent_dist.data * 0.5f).array().exp().rowwise().sum().matrix().segment(0,latent_dim);

			auto latent = sampler.grad_forward(latent_dist);
			auto reconstruction = decoder.grad_forward(latent);

			reconstruction.sync_cpu();

			Tensor d_cost = (reconstruction.data.array() - train_X[batch].data.array()).unaryExpr(&_sign);// l1 cost function, generates sharper images while l2 generates blurrier 

			d_cost.sync_gpu();

			cost += (reconstruction.data.array() - train_X[batch].data.array()).array().abs().mean();
			Tensor decoder_grad = decoder.compute_gradients(d_cost);
			Tensor sampler_grad = sampler.compute_gradients(decoder_grad);

			if(i % 100 == 0) {
				Tensor out = decoder.forward(fixed_noise);
				out.sync_cpu();
				Eigen::VectorXf buff = out.data.col(0);
				SDL_RenderClear(ren);
				draw_buff_rgb(ren,buff,image_w,UpscaleMode::NEAREST,0,0,256,256);
				buff = out.data.col(1);
				draw_buff_rgb(ren,buff,image_w,UpscaleMode::NEAREST,256,0,256,256);
				buff = out.data.col(2);
				draw_buff_rgb(ren,buff,image_w,UpscaleMode::NEAREST,0,256,256,256);
				buff = reconstruction.data.col(0);
				draw_buff_rgb(ren,buff,image_w,UpscaleMode::NEAREST,256,256,256,256);
				SDL_RenderPresent(ren);
				cout << "epoch " << epoch << " step " << i << "/" << batches << " cost=" << cost / (i + 1) << " elapsed time " << round(batch_time * 1000.0f) << " ms" << " beta=" << beta << " lr=" << lr << endl;
			}


			latent_dist.sync_cpu();

			Tensor kl_grad(latent_dist.dim(),latent_dist.batch_size());

			size_t h_dim = latent_dist.dim() / 2;

			for(int j = 0; j < latent_dist.batch_size(); j++) {
				kl_grad.data.col(j) << latent_dist.data.col(j).segment(0,h_dim),(latent_dist.data.col(j).segment(h_dim,h_dim).array().exp() - 1.0f) * 0.5f;
			}

			kl_grad.sync_gpu();
			Tensor cost_grad(sampler_grad.rows(),sampler_grad.cols(),true,0.0f);

			elemWiseSumA(kl_grad.gpu(),sampler_grad.gpu(),cost_grad.gpu(),beta,sampler_grad.size());

			encoder.compute_gradients(cost_grad);
			
			optim.step(false);
			grad_clip.step();
		}
		cout << "epoch " << epoch << " cost=" << cost / batches << " avg mu=" << latent_info.segment(0,latent_dim).sum() / (latent_dim * raw_train_X.size()) << " avg dev=" << latent_info.segment(latent_dim,latent_dim).sum() / (latent_dim * raw_train_X.size()) << endl;
		if(epoch % 5 == 0) {
			model.save_model("resdense-cifarcars-large.bin");
		}
		lr /= 1.05f;
	}
}