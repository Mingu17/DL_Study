#ifndef __COMMON_CONST_H__
#define __COMMON_CONST_H__

#include "xtensor/xmath.hpp"

namespace md {
	struct Constants {
		//static constexpr double PI = xt::numeric_constants<double>::PI;
		static constexpr float PI = 3.141592653589793238463f;
	};
}
#endif