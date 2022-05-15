#ifndef __MEAN_SQUARED_ERROR_H__
#define __MEAN_SQUARED_ERROR_H__

#include "../function.hpp"

namespace md {
	class MeanSquaredError : public Function {
	public:
		MeanSquaredError() {}

		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);
	};
}
#endif
