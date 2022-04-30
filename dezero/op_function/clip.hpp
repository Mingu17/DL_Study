#ifndef __CLIP_H__
#define __CLIP_H__

#include "../function.hpp"

namespace md {
	class Clip : public Function {
	public:
		Clip(double _x_min, double _x_max)
			: x_min(_x_min), x_max(_x_max) {
		}

		vec_spvar forward(vec_spvar& xs);
		vec_spvar backward(vec_spvar& gys);

	protected:
		double x_min;
		double x_max;
	};
}
#endif
