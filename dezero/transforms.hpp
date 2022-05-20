#ifndef __TRANSFORMS_H__
#define __TRANSFORMS_H__

#include "common.hpp"
#include "local_exception.hpp"

namespace md {
	class Transforms {
	public:
		Transforms() {}
		virtual xarr_f compute(const xarr_f& in) = 0;
	};
}
#endif
