#ifndef __BROADCAST_TO_H__
#define __BROADCAST_TO_H__

#include "../function.hpp"
#include "../utils.hpp"

namespace md {
	class BroadcastTo : public Function {
	public:
		BroadcastTo(const xarr_size& _shape) : shape(_shape) {
			param_reserve(1, 1, 1);
		}
		//vec_spvar forward(const vec_spvar& xs);
		//vec_spvar backward(const vec_spvar& gys);
		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);
	protected:
		xarr_size shape;
		xarr_size x_shape;
	};
}
#endif
