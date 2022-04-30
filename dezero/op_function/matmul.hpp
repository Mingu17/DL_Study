#ifndef __MATMUL_H__
#define __MATMUL_H__

#include "../function.hpp"

namespace md {
	class MatMul : public Function {
	public:
		MatMul() {}

		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);

		///// <summary>
		///// MatMul class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 2) {
		//		throw LocalException("(MatMul::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x = xs[0]->get_data();
		//		xarr_d& W = xs[1]->get_data();
		//		xarr_d res = xt::linalg::dot(x, W);
		//		return vec_spvar({ spvar::create(res) });
		//	}
		//}


		///// <summary>
		///// MatMul class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(MatMul::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		spvar& x = inputs[0];
		//		spvar& W = inputs[1];

		//		spvar gx = gy.dot(W.T());
		//		spvar gW = x.T().dot(gy);

		//		return vec_spvar({ gx, gW });
		//	}
		//}
	};
}
#endif
