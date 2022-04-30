#ifndef __ADD_H__
#define __ADD_H__

#include "../function.hpp"

namespace md {
	class Add : public Function {
	public:
		Add() {}
		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);
		
		///// <summary>
		///// Add class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 2) {
		//		throw LocalException("(Add::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x0 = xs[0]->get_data();
		//		xarr_d& x1 = xs[1]->get_data();
		//		x0_shape = x0.shape();
		//		x1_shape = x1.shape();
		//		xarr_d res = x0 + x1;
		//		return vec_spvar({ spvar::create(res) });
		//	}
		//}

		///// <summary>
		///// Add class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(Add::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		if (x0_shape != x1_shape) {
		//			spvar gx0 = gy.sum_to(x0_shape);
		//			spvar gx1 = gy.sum_to(x1_shape);
		//			return vec_spvar({ gx0, gx1 });
		//		}
		//		else {
		//			return vec_spvar({ gy, gy });
		//		}
		//	}
		//}
	protected:
		xarr_size x0_shape;
		xarr_size x1_shape;
	};
}
#endif