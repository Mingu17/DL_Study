#ifndef __TRANSFORMS_H__
#define __TRANSFORMS_H__

#include "common.hpp"
#include "local_exception.hpp"

namespace md {
	class Transforms {
	public:
		Transforms() {}
		virtual xarr_d compute(const xarr_d& in) = 0;
	};
}
#endif
