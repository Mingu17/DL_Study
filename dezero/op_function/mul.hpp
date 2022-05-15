#ifndef __MUL_H__
#define __MUL_H__

#include "../function.hpp"

namespace md {
	class Mul : public Function {
	public:
		Mul() {}

		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);
	};
}
#endif
