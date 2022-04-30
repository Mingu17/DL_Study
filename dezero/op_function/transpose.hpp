#ifndef __TRANSPOSE_H__
#define __TRANSPOSE_H__

#include "../function.hpp"

namespace md {
	class Transpose : public Function {
	public:
		Transpose() {}

		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);

		///// <summary>
		///// Transpose class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 1) {
		//		throw LocalException("(Transpose::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x = xs[0]->get_data();
		//		return vec_spvar({ spvar::create(xt::transpose(x)) });
		//	}
		//}

		///// <summary>
		///// Transpose class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(Transpose::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		return vec_spvar({ gy.transpose() });// transpose(gy)
		//	}
		//}
	};
}
#endif
