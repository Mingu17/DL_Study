#ifndef __HALF_SCALE_H__
#define __HALF_SCALE_H__

#include "../transforms.hpp"

namespace md {
	class HalfScale : public Transforms {
	public:
		HalfScale() {}
		xarr_d compute(const xarr_d& x) {
			return x * 0.5;
		}
	};
}
#endif