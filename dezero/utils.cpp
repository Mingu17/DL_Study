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

	void Utils::get_spiral(vec_xarr_d& out_data, bool train) {
		int seed = train ? 1984 : 2020;
		xt::random::seed(seed);

		int num_data = 100;
		int num_class = 3;
		int input_dim = 2;
		int data_size = num_class * num_data;
		xarr_d x_t = xt::zeros<double>({ data_size, input_dim });
		xarr_d x = xt::zeros<double>({ data_size, input_dim });
		xt::xarray<int> t_t = xt::zeros<int>({ data_size });
		xt::xarray<int> t = xt::zeros<int>({ data_size });

		for (int j = 0; j < num_class; ++j) {
			for (int i = 0; i < num_data; ++i) {
				double rate = static_cast<double>(i) / num_data;
				double radius = 1.0 * rate;
				xarr_d r_t = xt::random::randn<double>({ 1, 1 });
				double theta = 4.0 * j + rate * 4.0 + r_t(0) * 0.2;
				int ix = num_data * j + i;
				x_t(ix, 0) = radius * std::sin(theta);
				x_t(ix, 1) = radius * std::cos(theta);
				t_t(ix) = j;
			}
		}

		xt::xarray<int> indices = xt::random::permutation(num_data * num_class);
		for (int i = 0; i < indices.size(); ++i) {
			int idx = indices(i);
			x(i, 0) = x_t(idx, 0);
			x(i, 1) = x_t(idx, 1);
			t(i) = t_t(idx);
		}

		out_data.push_back(x);
		out_data.push_back(t);
	}
}