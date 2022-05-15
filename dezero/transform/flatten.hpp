#ifndef __FLATTEN_H__
#define __FLATTEN_H__

#include "../transforms.hpp"

namespace md {
	class Flatten : public Transforms {
	public:
		Flatten() {}
		xarr_d compute(const xarr_d& x) {
			xarr_size ori = x.shape();
			xarr_size target;
			int flat_size = 1;
			for (int i = 1; i < ori.size(); ++i) {
				flat_size = flat_size * ori[i];
			}
			target.push_back(ori[0]);
			target.push_back(flat_size);
			return const_cast<xarr_d&>(x).reshape(target);
		}
	};
}
#endif
