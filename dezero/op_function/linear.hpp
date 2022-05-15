#ifndef __LINEAR_H__
#define __LINEAR_H__

#include "../function.hpp"

namespace md {
	class Linear : public Function {
	public:
		Linear() {}

		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);
	};
}
#endif
