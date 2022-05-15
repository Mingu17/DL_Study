#ifndef __SPIRAL_H__
#define __SPIRAL_H__

#include "../datasets.hpp"
#include "../utils.hpp"

namespace md {
	class Spiral : public Dataset {
	public:
		Spiral(const bool _train = false,
			const vector<SP<Transforms>>& _transforms = {},
			const vector<SP<Transforms>>& _target_transforms = {})
			: Dataset(_train, _transforms, _target_transforms) {
			prepare();
		}

		void prepare() {
			xarr_d t_data, t_label;
			Utils::get_spiral(t_data, t_label);
			transform_data(t_data, t_label);
		}
	};
}
#endif