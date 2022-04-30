#include "utils.hpp"
#include "sp_variable.hpp"
#include "variable.hpp"

namespace md {
	xarr_d Utils::sum_to(xarr_d& x, xarr_size& shape) {
		int ndim = static_cast<int>(shape.size());
		int lead = static_cast<int>(x.dimension()) - ndim;
		int i = 0;
		xarr_size lead_axis;

		for (int i = 0; i < lead; ++i) {
			lead_axis.push_back(i);
		}

		for (int i = 0; i < shape.size(); ++i) {
			if (shape[i] == 1) {
				lead_axis.push_back(static_cast<size_t>(i) + lead);
			}
		}

		xarr_d y = xt::sum(x, lead_axis);

		if (lead > 0) {
			y = xt::squeeze(y);
		}
		return y;
	}

	spvar& Utils::reshape_sum_backward(spvar& gy, xarr_size& x_shape, xarr_size& axis, bool keepdims) {
		int ndim = static_cast<int>(x_shape.size());
		xarr_size tupled_axis = axis;
		if (axis.empty()) {
			tupled_axis.clear();
		}

		xarr_size shape = {};

		if (!(ndim == 0 || tupled_axis.empty() || keepdims)) {
			xarr_size actual_axis;
			for (int i = 0; i < tupled_axis.size(); ++i) {
				if (tupled_axis[i] >= 0) {
					actual_axis.push_back(tupled_axis[i]);
				}
				else {
					actual_axis.push_back(tupled_axis[i] + ndim);
				}
			}
			auto sorted = xt::sort(xt::adapt(actual_axis));
			shape = gy->get_shape();

			for (int i = 0; i < sorted.size(); ++i) {
				xarr_size::iterator iter = shape.begin();
				shape.insert(iter + sorted(i), 1);
			}
		}
		else {
			shape = gy->get_shape();
		}
		//return reshape(gy, shape);
		return gy.reshape(shape);
	}

	xarr_d Utils::logsumexp(xarr_d& x, size_t axis) {
		return logsumexp(x, xarr_size({ axis }));
	}

	xarr_d Utils::logsumexp(xarr_d& x, const xarr_size& axis) {
		xarr_d m = xt::amax(x, axis, xt::keep_dims);
		xarr_d y = x - m;
		xarr_d y_exp = xt::exp(y);
		xarr_d s = xt::sum(y_exp, axis, xt::keep_dims);
		xarr_d s_log = xt::log(s);
		m += s_log;
		return m;
	}
}