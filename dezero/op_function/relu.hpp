#ifndef __RELU_H__
#define __RELU_H__

#include "../function.hpp"

namespace md {
	class ReLU : public Function {
	public:
		ReLU() {}

		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);
	};
}
#endif
