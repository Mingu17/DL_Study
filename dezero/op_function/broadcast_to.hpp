#ifndef __BROADCAST_TO_H__
#define __BROADCAST_TO_H__

#include "../function.hpp"
#include "../utils.hpp"

namespace md {
	class BroadcastTo : public Function {
	public:
		BroadcastTo(xarr_size& _shape);
		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);

		///// <summary>
		///// BroadcastTo class constructor (Function)
		///// </summary>
		///// <param name="_shape"> - input shape variable</param>
		//BroadcastTo(xarr_size& _shape) : shape(_shape) {

		//}

		///// <summary>
		///// BroadcastTo class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 1) {
		//		throw LocalException("(BroadcastTo::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x = xs[0]->get_data();
		//		x_shape = x.shape();
		//		xarr_d res = xt::broadcast(x, shape);
		//		return vec_spvar({ spvar::create(res) });
		//	}
		//}

		///// <summary>
		///// BroadcastTo class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(BroadcastTo::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		return vec_spvar({ spvar::create(Utils::sum_to(gy->get_data(), x_shape)) });
		//	}
		//}
	protected:
		xarr_size shape;
		xarr_size x_shape;
	};
}
#endif
