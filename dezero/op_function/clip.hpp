#ifndef __CLIP_H__
#define __CLIP_H__

#include "../function.hpp"

namespace md {
	class Clip : public Function {
	public:
		Clip(const double _x_min, const double _x_max)
			: x_min(_x_min), x_max(_x_max) {
			param_reserve(1, 1, 1);
		}

		void forward(const vec_spvar& xs);
		void backward(const vec_spvar& gys);
	protected:
		double x_min;
		double x_max;
	};
}
#endif
