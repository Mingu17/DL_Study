#ifndef __SUB_H__
#define __SUB_H__

#include "../function.hpp"

namespace md {
	class Sub : public Function {
	public:
		Sub() {}

		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);
	};
}
#endif
