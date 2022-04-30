#ifndef __SUM_H__
#define __SUM_H__

#include "../function.hpp"
#include "../utils.hpp"

namespace md {
	class Sum : public Function {
	public:
		Sum(const xarr_size& _axis, bool _keepdims);
		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);
			
		///// <summary>
		///// Sum class constructor (Function)
		///// </summary>
		///// <param name="_axis"> - standard axis</param>
		///// <param name="_keepdims"> - keep dimensions</param>
		//Sum(xarr_size& _axis, bool _keepdims) : axis(_axis), keepdims(_keepdims) {

		//}

		///// <summary>
		///// Sum class forward function (Function)
		///// </summary>
		///// <param name="xs"> - input variable</param>
		///// <returns></returns>
		//vec_spvar forward(vec_spvar& xs) {
		//	if (xs.size() != 1) {
		//		throw LocalException("(Sum::forward) - Size mismatch");
		//	}
		//	else {
		//		xarr_d& x = xs[0]->get_data();
		//		x_shape = x.shape();
		//		if (axis.empty()) {
		//			return vec_spvar({ spvar::create(xt::sum(x)) });
		//		}
		//		else {
		//			xarr_d res = xt::sum(x, axis);
		//			return vec_spvar({ spvar::create(res) });
		//		}
		//	}
		//}

		///// <summary>
		///// Sum class backward function (Function)
		///// </summary>
		///// <param name="gys"> - gradient variable</param>
		///// <returns></returns>
		//vec_spvar backward(vec_spvar& gys) {
		//	if (gys.size() != 1) {
		//		throw LocalException("(Sum::backward) - Size mismatch");
		//	}
		//	else {
		//		spvar& gy = gys[0]->get_grad();
		//		spvar& gy_reshape = Utils::reshape_sum_backward(gy, x_shape, axis, keepdims);
		//		//spvar& gx = broadcast_to(gy_reshape, x_shape);
		//		spvar& gx = gy_reshape.broadcast_to(x_shape);
		//		return vec_spvar({ gx });
		//	}
		//}

	protected:
		xarr_size x_shape;
		xarr_size axis;
		bool keepdims;
	};
}
#endif
