#ifndef __RESHAPE_H__
#define __RESHAPE_H__

#include "../function.hpp"

namespace md {
	class Reshape : public Function {
	public:
		Reshape(xarr_size& s);
		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);
		
		///// <summary>
		///// Reshape class constructor (Function)
		///// </summary>
		///// <param name="s"> - target shape</param>
		//Reshape(xarr_size& s) : shape(s) {

		//}

		///// <summary>
		///// Reshape class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 1) {
		//		throw LocalException("(Reshape::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x = xs[0]->get_data();
		//		x_shape = x.shape();
		//		xarr_d res = x.reshape(shape);
		//		return vec_spvar({ spvar::create(res) });
		//	}
		//}

		///// <summary>
		///// Reshape class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(Reshape::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		//return vec_spvar({ reshape(gy, x_shape) });
		//		return vec_spvar({ gy.reshape(x_shape) });
		//	}
		//}
	protected:
		xarr_size shape;
		xarr_size x_shape;
	};
}
#endif
