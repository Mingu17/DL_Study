#ifndef __SUM_TO_H__
#define __SUM_TO_H__

#include "../function.hpp"
#include "../utils.hpp"

namespace md {
	class SumTo : public Function {
	public:
		SumTo(const xarr_size& _shape) : shape(_shape) {
			param_reserve(1, 1, 1);
		}

		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);

	protected:
		xarr_size shape;
		xarr_size x_shape;
	};
}
#endif
