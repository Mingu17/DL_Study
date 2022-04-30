#ifndef __FUNCTION_SET_H__
#define __FUNCTION_SET_H__

#include "op_function/add.hpp"
#include "op_function/mul.hpp"
#include "op_function/neg.hpp"
#include "op_function/sub.hpp"
#include "op_function/pow.hpp"
#include "op_function/div.hpp"
#include "op_function/sin.hpp"
#include "op_function/cos.hpp"
#include "op_function/tanh.hpp"
#include "op_function/exp.hpp"
#include "op_function/reshape.hpp"
#include "op_function/transpose.hpp"
#include "op_function/broadcast_to.hpp"
#include "op_function/sum_to.hpp"
#include "op_function/sum.hpp"
#include "op_function/matmul.hpp"
#include "op_function/mean_squared_error.hpp"
#include "op_function/linear.hpp"
#include "op_function/sigmoid.hpp"
#include "op_function/get_item.hpp"
#include "op_function/relu.hpp"
#include "op_function/softmax.hpp"
#include "op_function/softmax_cross_entropy.hpp"
#include "op_function/clip.hpp"

#include <memory>
#include <stack>

namespace md {
	extern std::stack<std::shared_ptr<Function>> op_stack;
	//extern int op_cnt;
	extern int start_cnt;
	extern int var_id;

	template<typename F, typename... Ts>
	static inline spvar& compute(Ts&... args) {
		op_stack.push(std::make_shared<F>());
		return op_stack.top()->call(args...)[0];
	}

	namespace op_func {
		template<typename T0, typename T1>
		static inline spvar& add(T0& x0, T1& x1) {
			return compute<Add>(x0, x1);
		}

		template<typename T0, typename T1>
		static inline spvar& mul(T0& x0, T1& x1) {
			return compute<Mul>(x0, x1);
		}

		template<typename T0, typename T1>
		static inline spvar& sub(T0& x0, T1& x1) {
			return compute<Sub>(x0, x1);
		}

		static inline spvar& neg(spvar& x) {
			return compute<Neg>(x);
		}

		template<typename T0, typename T1>
		static inline spvar& div(T0& x0, T1& x1) {
			return compute<Div>(x0, x1);
		}
	}

	namespace math {
		template<typename T0>
		static inline spvar& vpow(T0& x, const double c) {
			op_stack.push(std::make_shared<Pow>(c));
			return op_stack.top()->call(x)[0];
		}

		static inline spvar& vsin(spvar& x) {
			return compute<Sin>(x);
		}

		static inline spvar& vcos(spvar& x) {
			return compute<Cos>(x);
		}

		static inline spvar& vtanh(spvar& x) {
			return compute<Tanh>(x);
		}

		static inline spvar& vexp(spvar& x) {
			return compute<Exp>(x);
		}
	}

	namespace util_func {
		static inline spvar& vreshape(spvar& x, xarr_size target_size) {
			op_stack.push(std::make_shared<Reshape>(target_size));
			return op_stack.top()->call(x)[0];
		}

		static inline spvar& vtranspose(spvar& x) {
			return compute<Transpose>(x);
		}

		static inline spvar& vbroadcast_to(spvar& x, xarr_size& shape) {
			if (x->get_data().shape() == shape) {
				return x;
			}
			else {
				op_stack.push(std::make_shared<BroadcastTo>(shape));
				return op_stack.top()->call(x)[0];
			}
		}

		static inline spvar& vsum_to(spvar& x, xarr_size& shape) {
			if (x->get_data().shape() == shape) {
				return x;
			}
			else {
				op_stack.push(std::make_shared<SumTo>(shape));
				return op_stack.top()->call(x)[0];
			}
		}

		static inline spvar& vsum(spvar& x, const xarr_size& axis, bool keepdims = false) {
			op_stack.push(std::make_shared<Sum>(axis, keepdims));
			return op_stack.top()->call(x)[0];
		}

		static inline spvar& vdot(spvar& x, spvar& W) {
			return compute<MatMul>(x, W);
		}

		static inline spvar& vmean_squared_error(spvar& x0, spvar& x1) {
			return compute<MeanSquaredError>(x0, x1);
		}

		static inline spvar& vlinear(spvar& x, spvar& W) {
			return compute<Linear>(x, W);
		}

		static inline spvar& vlinear(spvar& x, spvar& W, spvar& b) {
			return compute<Linear>(x, W, b);
		}

		static inline spvar& vsigmoid(spvar& x) {
			return compute<Sigmoid>(x);
		}

		static inline spvar& vsoftmax(spvar& x, const xarr_size& axis) {
			op_stack.push(std::make_shared<Softmax>(axis));
			return op_stack.top()->call(x)[0];
		}

		static inline spvar& vrelu(spvar& x) {
			return compute<ReLU>(x);
		}

		static inline spvar& vsoftmax_cross_entropy(spvar& x, spvar& t) {
			op_stack.push(std::make_shared<SoftmaxCrossEntropy>());
			return op_stack.top()->call(x, t)[0];
		}

		static inline spvar& vget_item(spvar& x, const vec_xslice& slices) {
			op_stack.push(std::make_shared<GetItem>(slices));
			return op_stack.top()->call(x)[0];
		}
		
		static inline spvar& vclip(spvar& x, double x_min, double x_max) {
			op_stack.push(std::make_shared<Clip>(x_min, x_max));
			return op_stack.top()->call(x)[0];
		}
	}

	static inline void start_op() {
		if (start_cnt == -1) {
			start_cnt = static_cast<int>(op_stack.size());
		}
		else {
			throw LocalException("(start_op) - not initialize state");
		}
	}

	static inline void clear_op(bool isDebug = false) {
		int now_cnt = static_cast<int>(op_stack.size());
		int op = now_cnt - start_cnt;
		for (int i = 0; i < op; ++i) 
			op_stack.pop();
		if (isDebug) {
			std::cout << "clear " << op << " variables." << std::endl;
		}
		//start_cnt = -1;
	}
}
#endif
