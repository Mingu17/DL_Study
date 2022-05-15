#ifndef __CLIP_H__
#define __CLIP_H__

#include "../function.hpp"

namespace md {
	class Clip : public Function {
	public:
		Clip(const double _x_min, const double _x_max)
			: x_min(_x_min), x_max(_x_max) {
		}

		vec_spvar forward(const vec_spvar& xs);
		vec_spvar backward(const vec_spvar& gys);

	protected:
		double x_min;
		double x_max;
	};
}
#endif
