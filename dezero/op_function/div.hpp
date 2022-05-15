#ifndef __DIV_H__
#define __DIV_H__

#include "../function.hpp"

namespace md {
	class Div : public Function {
	public:
		Div() {}

		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);
	};
}
#endif
