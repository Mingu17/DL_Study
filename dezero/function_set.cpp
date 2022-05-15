#include "function_set.hpp"
#include "sp_variable.hpp"
#include "variable.hpp"

namespace md {
	spvar& util_func::broadcast_to(spvar& x, const xarr_size& shape) {
		if (x->get_data().shape() == shape) {
			return x;
		}
		else {
			op_stack.push(std::make_shared<BroadcastTo>(shape));
			return op_stack.top()->call(x)[0];
		}
	}

	spvar& util_func::sum_to(spvar& x, const xarr_size& shape) {
		if (x->get_data().shape() == shape) {
			return x;
		}
		else {
			op_stack.push(std::make_shared<SumTo>(shape));
			return op_stack.top()->call(x)[0];
		}
	}

	spvar util_func::accuracy(const spvar& y, const spvar& t) {
		xarr_d& ty = y->get_data();
		xarr_d& tt = t->get_data();
		xarr_d t_pred = xt::argmax(ty, 1);
		xarr_d pred = t_pred.reshape(tt.shape());
		xarr_d result = xt::equal(pred, tt);
		return spvar::create(xt::mean(result));
	}
}