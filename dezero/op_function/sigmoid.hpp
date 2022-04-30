#ifndef __SIGMOID_H__
#define __SIGMOID_H__

#include "../function.hpp"

namespace md {
	class Sigmoid : public Function {
	public:
		Sigmoid() {}

		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);

		///// <summary>
		///// Sigmoid class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 1) {
		//		throw LocalException("(Sigmoid::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x = xs[0]->get_data();
		//		xarr_d res = xt::tanh(x * 0.5) * 0.5 + 0.5;
		//		return vec_spvar({ spvar::create(res) });
		//	}
		//}

		///// <summary>
		///// Sigmoid class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(Sigmoid::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		spvar& y = outputs[0];
		//		return vec_spvar({ gy * y * (1.0 - y) });
		//	}
		//}
	};
}
#endif
