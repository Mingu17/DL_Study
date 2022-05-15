#ifndef __SUM_H__
#define __SUM_H__

#include "../function.hpp"
#include "../utils.hpp"

namespace md {
	class Sum : public Function {
	public:
		Sum(const xarr_size& _axis, bool _keepdims);
		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);
			
	protected:
		xarr_size x_shape;
		xarr_size axis;
		bool keepdims;
	};
}
#endif
