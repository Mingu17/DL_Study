#ifndef __HALF_SCALE_H__
#define __HALF_SCALE_H__

#include "../transforms.hpp"

namespace md {
	class HalfScale : public Transforms {
	public:
		HalfScale() {}
		xarr_f compute(const xarr_f& x) override {
			return x * 0.5f;
		}
	};
}
#endif