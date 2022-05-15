#ifndef __SUM_TO_H__
#define __SUM_TO_H__

#include "../function.hpp"
#include "../utils.hpp"

namespace md {
	class SumTo : public Function {
	public:
		SumTo(const xarr_size& _shape);
		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);
		
	protected:
		xarr_size shape;
		xarr_size x_shape;
	};
}
#endif
