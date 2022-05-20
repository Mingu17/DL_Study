#ifndef __FLATTEN_H__
#define __FLATTEN_H__

#include "../transforms.hpp"

namespace md {
	class Flatten : public Transforms {
	public:
		Flatten() {}
		xarr_f compute(const xarr_f& x) override {
			xarr_size ori = x.shape();
			xarr_size target;
			size_t flat_size = 1;
			for (size_t i = 1; i < ori.size(); ++i) {
				flat_size = flat_size * ori[i];
			}
			target.push_back(ori[0]);
			target.push_back(flat_size);
			return const_cast<xarr_f&>(x).reshape(target);
		}
	};
}
#endif
