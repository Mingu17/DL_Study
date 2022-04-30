#include "sgd.hpp"
#include "momentum_sgd.hpp"
#include "adagrad.hpp"
#include "adadelta.hpp"
#include "adam.hpp"
#include "../sp_variable.hpp"
#include "../variable.hpp"

namespace md {
	void SGD::update_one(const parameter& param) {
		param->get_data() -= lr * param->get_grad()->get_data();
	}

	void MomentumSGD::update_one(const parameter& param) {
		ull id = param.get_id();
		if (vs.find(id) == vs.end()) {
			xarr_d zd = xt::zeros_like(param->get_data());
			vs.insert(std::make_pair(id, zd));
		}
		xarr_d& v = vs[id];
		v *= momentum;
		v -= lr * param->get_grad()->get_data();
		param->get_data() += v;
	}

	void AdaGrad::update_one(const parameter& param) {
		ull id = param.get_id();
		if (hs.find(id) == hs.end()) {
			hs.insert(std::make_pair(id, xt::zeros_like(param->get_data())));
		}
		xarr_d& h = hs[id];
		xarr_d& grad = param->get_grad()->get_data();
		h += grad * grad;
		param->get_data() -= (lr * grad / (xt::sqrt(h) + eps));
	}

	void AdaDelta::update_one(const parameter& param) {
		ull id = param.get_id();
		if (msg.find(id) == msg.end()) {
			msg.insert(std::make_pair(id, xt::zeros_like(param->get_data())));
			msdx.insert(std::make_pair(id, xt::zeros_like(param->get_data())));
		}

		xarr_d& g = msg[id];
		xarr_d& x = msdx[id];
		xarr_d& grad = param->get_grad()->get_data();

		g *= rho;
		g += ((1.0 - rho) * grad * grad);
		xarr_d dx = xt::sqrt((x + eps) / (g + eps)) * grad;
		x *= rho;
		x += ((1.0 - rho) * dx * dx);
		param->get_data() -= dx;
	}

	void Adam::update() {
		t += 1;
		Optimizer::update();
	}

	void Adam::update_one(const parameter& param) {
		ull id = param.get_id();
		if (ms.find(id) == ms.end()) {
			ms.insert(std::make_pair(id, xt::zeros_like(param->get_data())));
			vs.insert(std::make_pair(id, xt::zeros_like(param->get_data())));
		}
		xarr_d& m = ms[id];
		xarr_d& v = vs[id];
		xarr_d& grad = param->get_grad()->get_data();

		m += (1.0 - beta1) * (grad - m);
		v += (1.0 - beta2) * (grad * grad - v);
		param->get_data() -= get_lr() * m / (xt::sqrt(v) + eps);
	}

	double Adam::get_lr() {
		double fix1 = 1.0 - std::pow(beta1, t);
		double fix2 = 1.0 - std::pow(beta2, t);
		return alpha * std::sqrt(fix2) / fix1;
	}
}