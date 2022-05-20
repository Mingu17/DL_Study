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

	std::ostream& operator<<(std::ostream& out, const spvar& v) {
		out << /*xt::print_options::precision(8) <<*/ v->get_data();
		return out;
	}

	spvar& spvar::operator+(const spvar& x) {
		return md::op_func::add(*this, x);
	}

	spvar& spvar::operator+(const float& d) {
		return md::op_func::add(*this, d);
	}

	spvar& operator+(const float& x0, const spvar& x1) {
		return md::op_func::add(x0, x1);
	}

	spvar& spvar::operator*(const spvar& x) {
		return md::op_func::mul(*this, x);
	}

	spvar& spvar::operator*(const float& d) {
		return md::op_func::mul(*this, d);
	}

	spvar& operator*(const float& x0, const spvar& x1) {
		return md::op_func::mul(x0, x1);
	}

	spvar& spvar::operator-(const spvar& x) {
		return md::op_func::sub(*this, x);
	}

	spvar& spvar::operator-(const float& d) {
		return md::op_func::sub(*this, d);
	}

	spvar& operator-(const float& x0, const spvar& x1) {
		return md::op_func::sub(x0, x1);
	}

	spvar& spvar::operator-() {
		return md::op_func::neg(*this);
	}

	spvar& spvar::operator/(const spvar& x) {
		return md::op_func::div(*this, x);
	}

	spvar& spvar::operator/(const float& d) {
		return md::op_func::div(*this, d);
	}

	spvar& operator/(const float& x0, const spvar& x1) {
		return md::op_func::div(x0, x1);
	}

	double spvar::operator[](int idx) {
		return ptr->get_data()(idx);
	}

	spvar& spvar::reshape(const xarr_size& target_size) {
		return md::inner_util_func::vreshape(*this, target_size);
	}

	spvar& spvar::transpose() {
		return md::inner_util_func::vtranspose(*this);
	}

	spvar& spvar::T() {
		return md::inner_util_func::vtranspose(*this);
	}

	spvar& spvar::dot(const spvar& W) {
		return md::inner_util_func::vdot(*this, W);
	}

	spvar& spvar::matmul(const spvar& x) { //equal dot()
		return md::inner_util_func::vdot(*this, x);
	}

	spvar& spvar::get_item(const vec_xslice& slices) {
		return md::inner_util_func::vget_item(*this, slices);
	}

	spvar& spvar::clip(const float x_min, const float x_max) {
		return md::inner_util_func::vclip(*this, x_min, x_max);
	}

	void spvar::setup_id() {
		id = var_id;
		var_id++;
	}

	////create function
	spvar spvar::create(const float& in) {
		return spvar(new Variable(in));
	}

	spvar spvar::create(const xarr_f& in) {
		return spvar(new Variable(in));
	}

	spvar spvar::create(const Variable& in) {
		return spvar(new Variable(in));
	}
}