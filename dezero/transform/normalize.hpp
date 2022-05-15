#ifndef __NORMALIZE_H__
#define __NORMALIZE_H__

#include "../transforms.hpp"

namespace md {
	class Normalize : public Transforms {
	public:
		Normalize(const double _mean = 0.0, const double _std = 1.0)
			: isscalar_mean(true), isscalar_std(true) {
			init(xarr_d({ _mean }), xarr_d({ _std }));
		}

		Normalize(const xarr_d& _mean = xarr_d({ 0.0 }),
			const xarr_d& _std = xarr_d({ 1.0 })) 
			: isscalar_mean(false), isscalar_std(false) {
			init(_mean, _std);
		}

		xarr_d compute(const xarr_d& x) {
			xarr_d t_mean = mean;
			xarr_d t_std = std;

			if (!isscalar_mean) {
				xarr_i mshape = xt::ones<int>({ x.dimension() });
				mshape(0) = (mean.shape()[0] == 1) ? x.shape()[0] : mean.shape()[0];
				t_mean = mean.reshape(mshape);
			}
			if (!isscalar_std) {
				xarr_i rshape = xt::ones<int>({ x.dimension() });
				rshape(0) = (std.shape()[0] == 1) ? x.shape()[0] : std.shape()[0];
				t_std = std.reshape(rshape);
			}
			return (x - t_mean) / t_std;
		}

	protected:
		void init(const xarr_d& _mean, const xarr_d& _std) {
			mean = _mean;
			std = _std;
		}
	protected:
		xarr_d mean;
		xarr_d std;
		bool isscalar_mean;
		bool isscalar_std;
	};
}
#endif
