#ifndef __LINEAR_H__
#define __LINEAR_H__

#include "../function.hpp"

namespace md {
	class Linear : public Function {
	public:
		Linear() {}

		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);
		
		///// <summary>
		///// Linear class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (!(xs.size() == 2 || xs.size() == 3)) {
		//		throw LocalException("(Linear::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x = xs[0]->get_data();
		//		xarr_d& W = xs[1]->get_data();
		//		xarr_d y = xt::linalg::dot(x, W);

		//		if (xs.size() == 2) {
		//			return vec_spvar({ spvar::create(y) });
		//		}
		//		else {
		//			xarr_d& b = xs[2]->get_data();
		//			return vec_spvar({ spvar::create(y + b) });
		//		}
		//	}
		//}

		///// <summary>
		///// Linear class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(Linear::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		spvar& x = inputs[0];
		//		spvar& W = inputs[1];

		//		spvar gx = gy.matmul(W.T());
		//		spvar gW = x.T().matmul(gy);

		//		if (inputs.size() == 2) {
		//			return vec_spvar({ gx, gW });
		//		}
		//		else {
		//			spvar& b = inputs[2];
		//			spvar gb = gy.sum_to(const_cast<xarr_size&>(b->get_shape()));
		//			return vec_spvar({ gx, gW, gb });
		//		}
		//	}
		//}
	};
}
#endif
