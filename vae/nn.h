#pragma once
#include <math.h>
#include <assert.h>
#include <vector>
#include <random>
#include <Eigen/Eigen>

namespace nn_rand {
	std::mt19937 global_rng{};
}

typedef double(*xy_func)(double);

double _relu(double x) {
	return std::max(x,0.0);
}
double d_relu(double x) {
	return x >= 0 ? 1 : 0;
}
#define K_LEAKY 0.001 // leaky relu coefficient
double _lrelu(double x) {
	return std::max(x,K_LEAKY * x);
}

double d_lrelu(double x) {
	return x > 0 ? 1 : K_LEAKY;
}
double _linear(double x) {
	return x;
}
double d_linear(double x) {
	return 1;
}

double _tanh(double x) {
	return std::tanh(x);
}

double d_tanh(double y) {
	return (1 - y * y);
}

double x_d_tanh(double x) {
	return (1 - std::pow(std::tanh(x),2));
}

double _sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

double d_sigmoid(double y)
{
	return y * (1.0f - y);
}

double _sign(double x) {
	return (!signbit(x)) * 2.0 - 1.0;
}

class deriv_func {
public:
	xy_func f,deriv;
	__forceinline double operator()(const double x) {
		return f(x);
	}
	__forceinline Eigen::VectorXd operator()(const Eigen::VectorXd& x) {
		return x.unaryExpr(f);
	}
};

namespace act {
	const deriv_func relu{_relu,d_relu};
	const deriv_func linear{_linear,d_linear};
	const deriv_func tanh{std::tanh,x_d_tanh};
	const deriv_func leaky_relu{_lrelu,d_lrelu};
}

class Pass {
public:
	virtual Eigen::VectorXd forward(const Eigen::VectorXd& in) = 0;
	virtual Eigen::VectorXd grad_forward(const Eigen::VectorXd& in) = 0;
	virtual void reset_gradients() = 0;
	virtual void apply_gradients(double lr) = 0;
	virtual Eigen::VectorXd compute_gradients(const Eigen::VectorXd& flow_back) = 0;  // returns the partial derivative of the cost in respect to input vector
};

class Dense : public Pass {
public:
	Eigen::VectorXd linear;
	Eigen::VectorXd in_activation;

	Eigen::MatrixXd weights;
	Eigen::VectorXd biases;

	Eigen::MatrixXd w_grad_mean;
	Eigen::VectorXd b_grad_mean;

	size_t grad_samples;

	deriv_func activ_f;

	double lambda_l2 = 0;

	Dense(size_t in_dim,size_t out_dim,double Lambda_l2 = 0,bool use_he_init = true,deriv_func activation = act::linear):activ_f(activation),lambda_l2(Lambda_l2) {
		weights = Eigen::MatrixXd(out_dim,in_dim);
		biases = Eigen::VectorXd(out_dim);
		if(!use_he_init) {
			weights.setRandom();
		}
		else {
			std::normal_distribution<double> dist(0,sqrt(2.0 / in_dim));
			for(int i = 0; i < weights.cols(); i++) {
				for(int j = 0; j < weights.rows(); j++) {
					weights(j,i) = dist(nn_rand::global_rng);
				}
			}
		}
		biases.setRandom();
		w_grad_mean = weights;
		b_grad_mean = biases;
		reset_gradients();
	}
	inline Eigen::VectorXd forward(const Eigen::VectorXd& in) override {
		return activ_f(weights * in + biases);
	}
	inline Eigen::VectorXd grad_forward(const Eigen::VectorXd& in) override {
		in_activation = in;
		linear = weights * in + biases;
		return activ_f(linear);
	}
	inline void reset_gradients() override {
		w_grad_mean.setZero();
		b_grad_mean.setZero();
		grad_samples = 0;
	}
	void apply_gradients(double lr) override {
		if(grad_samples > 0) {
			weights -= lr * w_grad_mean;
			biases -= lr * b_grad_mean;
			reset_gradients();
		}
	}
	Eigen::VectorXd compute_gradients(const Eigen::VectorXd& flow_back) override { // returns the partial derivative of the cost in respect to input vector
		assert(flow_back.size() == biases.size()); // "flow_back partial gradients vector dim must match out_dim";
		Eigen::VectorXd pd_act_lin = linear.unaryExpr(activ_f.deriv);
		Eigen::VectorXd pd_fb_lin = pd_act_lin.array() * flow_back.array();


		grad_samples++;

		double alpha = 1.0 / grad_samples;

		w_grad_mean += alpha * ((pd_fb_lin * in_activation.transpose() + lambda_l2 * 2 * (weights)) - w_grad_mean);
		b_grad_mean += alpha * (pd_fb_lin - b_grad_mean);


		Eigen::VectorXd ret_flow_back;
		ret_flow_back.resize(weights.cols());

		for(int i = 0; i < weights.cols(); i++) {
			ret_flow_back(i) = (pd_fb_lin.array() * weights.col(i).array()).sum();
		}

		return ret_flow_back;
	}
};

class Softmax : public Pass {
	Eigen::VectorXd activation;
	Eigen::VectorXd forward(const Eigen::VectorXd& in) override {
		Eigen::ArrayXd ex = (in.array() - in.maxCoeff()).exp(); // avoids overflow without changing output since
		return (ex / ex.sum());;
	}
	Eigen::VectorXd grad_forward(const Eigen::VectorXd& in) override {
		activation = forward(in);
		return activation;
	}
	void reset_gradients() override {};
	void apply_gradients(double lr) override {};
	Eigen::VectorXd compute_gradients(const Eigen::VectorXd& flow_back) override {
		double dot = activation.dot(flow_back);
		return activation.array() * (flow_back.array() - dot);
	}
};

class ReLU : public Pass {
	Eigen::VectorXd in_activation;
	Eigen::VectorXd forward(const Eigen::VectorXd& in) override {
		return in.unaryExpr(&_relu);
	}
	Eigen::VectorXd grad_forward(const Eigen::VectorXd& in) override {
		in_activation = in;
		return forward(in);
	}
	void reset_gradients() override {};
	void apply_gradients(double lr) override {};
	Eigen::VectorXd compute_gradients(const Eigen::VectorXd& flow_back) override {
		return flow_back.array() * in_activation.unaryExpr(&d_relu).array();
	}
};

class Tanh : public Pass {
	Eigen::VectorXd activation;
	Eigen::VectorXd forward(const Eigen::VectorXd& in) override {
		return in.unaryExpr(&_tanh);
	}
	Eigen::VectorXd grad_forward(const Eigen::VectorXd& in) override {
		activation = forward(in);
		return activation;
	}
	void reset_gradients() override {};
	void apply_gradients(double lr) override {};
	Eigen::VectorXd compute_gradients(const Eigen::VectorXd& flow_back) override {
		return flow_back.array() * activation.unaryExpr(&d_tanh).array();
	}
};

class Sigmoid : public Pass {
	Eigen::VectorXd activation;
	Eigen::VectorXd forward(const Eigen::VectorXd& in) override {
		return in.unaryExpr(&_sigmoid);
	}
	Eigen::VectorXd grad_forward(const Eigen::VectorXd& in) override {
		activation = forward(in);
		return activation;
	}
	void reset_gradients() override {};
	void apply_gradients(double lr) override {};
	Eigen::VectorXd compute_gradients(const Eigen::VectorXd& flow_back) override {
		return flow_back.array() * activation.unaryExpr(&d_sigmoid).array();
	}
};

class LeakyReLU : public Pass {
	Eigen::VectorXd in_activation;
	double coeff = 0;
public:
	LeakyReLU(double Coeff): coeff(Coeff) {};
	Eigen::VectorXd forward(const Eigen::VectorXd& in) override {
		return in.cwiseMax(in * coeff);
	}
	Eigen::VectorXd grad_forward(const Eigen::VectorXd& in) override {
		in_activation = in;
		return forward(in);
	}
	void reset_gradients() override {};
	void apply_gradients(double lr) override {};
	Eigen::VectorXd compute_gradients(const Eigen::VectorXd& flow_back) override {
		return flow_back.array() * in_activation.unaryExpr([this](double x) { return x >= 0.0 ? 1.0 : coeff; }).array();
	}
};

class Dropout : public Pass {
public:
	double prob; // probability of a neuron to be set to zero (influence only grad_forward)
	Eigen::VectorXd mask;
	Dropout(double Probability):prob(Probability) {};
	Eigen::VectorXd forward(const Eigen::VectorXd& in) override {
		return in;
	}
	Eigen::VectorXd grad_forward(const Eigen::VectorXd& in) override {
		mask.setRandom(in.size());
		mask = mask.unaryExpr([this](double x) { return (((x + 1) * 0.5) < prob ? 0.0 : 1.0); });
		return mask.array() * in.array();
	}
	void reset_gradients() override {};
	void apply_gradients(double lr) override {};
	Eigen::VectorXd compute_gradients(const Eigen::VectorXd& flow_back) override {
		return flow_back.array() * mask.array();
	}
};

class Sequential {
public:
	std::vector<Pass*> layers;
	Eigen::VectorXd forward(const Eigen::VectorXd& in) {
		Eigen::VectorXd pass = in;
		for(auto l : layers) {
			pass = l->forward(pass);
		}
		return pass;
	}
	Eigen::VectorXd grad_forward(const Eigen::VectorXd& in) {
		Eigen::VectorXd pass = in;
		for(auto l : layers) {
			pass = l->grad_forward(pass);
		}
		return pass;
	}
	Eigen::VectorXd compute_gradients(const Eigen::VectorXd& flow_back) {
		Eigen::VectorXd _flow_back = flow_back;
		for(int i = layers.size() - 1; i >= 0; i--) {
			_flow_back = layers[i]->compute_gradients(_flow_back);
		}
		return _flow_back;
	}
	void reset_gradients() {
		for(auto l : layers) {
			l->reset_gradients();
		}
	}
	void apply_gradients(double lr) {
		for(auto l : layers) {
			l->apply_gradients(lr);
		}
	}
};