#ifndef __SIN_H__
#define __SIN_H__

#include "../function.hpp"

namespace md {
	class Sin : public Function {
	public:
		Sin() {}

		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);
	};
}
#endif