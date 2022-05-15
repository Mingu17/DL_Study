#ifndef __POW_H__
#define __POW_H__

#include "../function.hpp"

namespace md {
	class Pow : public Function {
	public:
		Pow(double _c);
		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);

	protected:
		double c;
	};
}
#endif
