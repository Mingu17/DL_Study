#ifndef __SUM_TO_H__
#define __SUM_TO_H__

#include "../function.hpp"
#include "../utils.hpp"

namespace md {
	class SumTo : public Function {
	public:
		SumTo(xarr_size& _shape);
		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);
		
		///// <summary>
		///// SumTo class constructor
		///// </summary>
		///// <param name="_shape"> - shape of sum?</param>
		//SumTo(xarr_size& _shape) : shape(_shape) {

		//}

		///// <summary>
		///// SumTo class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 1) {
		//		throw LocalException("(SumTo::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x = xs[0]->get_data();
		//		x_shape = x.shape();
		//		xarr_d res = Utils::sum_to(x, shape);
		//		return vec_spvar({ spvar::create(res) });
		//	}
		//}

		///// <summary>
		///// SumTo class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(SumTo::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		return vec_spvar({ gy.broadcast_to(x_shape) });
		//	}
		//}
	protected:
		xarr_size shape;
		xarr_size x_shape;
	};
}
#endif
