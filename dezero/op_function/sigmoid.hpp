#ifndef __SIGMOID_H__
#define __SIGMOID_H__

#include "../function.hpp"

namespace md {
	class Sigmoid : public Function {
	public:
		Sigmoid() {}

		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);
	};
}
#endif
