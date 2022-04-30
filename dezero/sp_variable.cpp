#include "sp_variable.hpp"
#include "variable.hpp"
#include "function_set.hpp"
#include <stack>
#include "utils.hpp"

namespace md {
	std::stack<std::shared_ptr<Function>> op_stack;
	//int op_cnt = 0;
	int start_cnt = -1;
	int var_id = 1982;

	std::ostream& operator<<(std::ostream& out, spvar& v) {
		out << xt::print_options::precision(16) << v->get_data();
		return out;
	}

	spvar& spvar::operator+(const spvar& x) {
		return md::op_func::add(*this, x);
	}

	spvar& spvar::operator+(const double& d) {
		return md::op_func::add(*this, d);
	}

	spvar& operator+(const double& x0, const spvar& x1) {
		return md::op_func::add(x0, x1);
	}

	spvar& spvar::operator*(const spvar& x) {
		return md::op_func::mul(*this, x);
	}

	spvar& spvar::operator*(const double& d) {
		return md::op_func::mul(*this, d);
	}

	spvar& operator*(const double& x0, const spvar& x1) {
		return md::op_func::mul(x0, x1);
	}

	spvar& spvar::operator-(const spvar& x) {
		return md::op_func::sub(*this, x);
	}

	spvar& spvar::operator-(const double& d) {
		return md::op_func::sub(*this, d);
	}

	spvar& operator-(const double& x0, const spvar& x1) {
		return md::op_func::sub(x0, x1);
	}

	spvar& spvar::operator-() {
		return md::op_func::neg(*this);
	}

	spvar& spvar::operator/(const spvar& x) {
		return md::op_func::div(*this, x);
	}

	spvar& spvar::operator/(const double& d) {
		return md::op_func::div(*this, d);
	}

	spvar& operator/(const double& x0, const spvar& x1) {
		return md::op_func::div(x0, x1);
	}

	spvar& spvar::pow(const double c) {
		return md::math::vpow(*this, c);
	}

	spvar& spvar::sin() {
		return md::math::vsin(*this);
	}

	spvar& spvar::cos() {
		return md::math::vcos(*this);
	}

	spvar& spvar::tanh() {
		return md::math::vtanh(*this);
	}

	spvar& spvar::exp() {
		return md::math::vexp(*this);
	}

	spvar& spvar::reshape(xarr_size target_size) {
		return md::util_func::vreshape(*this, target_size);
	}

	spvar& spvar::transpose() {
		return md::util_func::vtranspose(*this);
	}

	spvar& spvar::T() {
		return md::util_func::vtranspose(*this);
	}

	spvar& spvar::broadcast_to(xarr_size& shape) {
		return md::util_func::vbroadcast_to(*this, shape);
	}

	spvar& spvar::sum_to(xarr_size& shape) {
		return md::util_func::vsum_to(*this, shape);
	}

	spvar& spvar::sum(const xarr_size& axis, bool keepdims) {
		return md::util_func::vsum(*this, axis, keepdims);
	}

	spvar& spvar::dot(spvar& W) {
		return md::util_func::vdot(*this, W);
	}

	spvar& spvar::matmul(spvar& x) { //equal dot()
		return md::util_func::vdot(*this, x);
	}

	spvar& spvar::mean_squared_error(spvar& x0) {
		return md::util_func::vmean_squared_error(*this, x0);
	}

	spvar& spvar::softmax(size_t axis) {
		return softmax(xarr_size({ axis }));
	}

	spvar& spvar::softmax(const xarr_size& axis) {
		return md::util_func::vsoftmax(*this, axis);
	}

	spvar& spvar::softmax_cross_entropy(spvar& t) {
		return md::util_func::vsoftmax_cross_entropy(*this, t);
	}

	spvar& spvar::relu() {
		return md::util_func::vrelu(*this);
	}

	spvar& spvar::linear(spvar& W, spvar& b) {
		return md::util_func::vlinear(*this, W, b);
	}

	spvar& spvar::linear(spvar& W, spvar&& b) {
		if (b == nullptr) {
			return md::util_func::vlinear(*this, W);
		}
		else {
			return md::util_func::vlinear(*this, W, b);
		}
	}

	spvar& spvar::linear_simple(spvar& W, spvar&& b) {
		spvar& t = this->matmul(W);
		if (b == nullptr) {
			return t;
		}
		else {
			spvar& y = t + b;
			t.release();
			return y;
		}
	}

	spvar& spvar::sigmoid_simple() {
		spvar& y = 1.0 / (1.0 + md::math::vexp(*this));
		return y;
	}

	spvar& spvar::sigmoid() {
		return md::util_func::vsigmoid(*this);
	}

	spvar& spvar::get_item(const vec_xslice& slices) {
		return md::util_func::vget_item(*this, slices);
	}

	spvar& spvar::clip(double x_min, double x_max) {
		return md::util_func::vclip(*this, x_min, x_max);
	}

	void spvar::setup_id() {
		id = var_id;
		var_id++;
	}

	////create function
	spvar spvar::create(const double& in) {
		return spvar(new Variable(in));
	}

	spvar spvar::create(const xarr_d& in) {
		return spvar(new Variable(in));
	}

	spvar spvar::create(const Variable& in) {
		return spvar(new Variable(in));
	}
}