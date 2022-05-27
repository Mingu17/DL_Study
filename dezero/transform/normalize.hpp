#ifndef __NORMALIZE_H__
#define __NORMALIZE_H__

#include "../transforms.hpp"

namespace md {
	class Normalize : public Transforms {
	public:
		Normalize(const float _mean = 0.0f, const float _std = 1.0f)
			: isscalar_mean(true), isscalar_std(true) {
			init(xarr_f({ _mean }), xarr_f({ _std }));
		}

		Normalize(const xarr_f& _mean = xarr_f({ 0.0f }),
			const xarr_f& _std = xarr_f({ 1.0f })) 
			: isscalar_mean(false), isscalar_std(false) {
			init(_mean, _std);
		}

		xarr_f compute(const xarr_f& x) override{
			xarr_f t_mean = mean;
			xarr_f t_std = std;

			if (!isscalar_mean) {
				xarr_st mshape = xt::ones<size_t>({ x.dimension() });
				mshape(0) = (mean.shape()[0] == 1) ? x.shape()[0] : mean.shape()[0];
				t_mean = mean.reshape(mshape);
			}
			if (!isscalar_std) {
				xarr_st rshape = xt::ones<size_t>({ x.dimension() });
				rshape(0) = (std.shape()[0] == 1) ? x.shape()[0] : std.shape()[0];
				t_std = std.reshape(rshape);
			}
			return (x - t_mean) / t_std;
		}

	protected:
		void init(const xarr_f& _mean, const xarr_f& _std) {
			mean = _mean;
			std = _std;
		}
	protected:
		xarr_f mean;
		xarr_f std;
		bool isscalar_mean;
		bool isscalar_std;
	};
}
#endif
