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
#include "op_function/get_item_grad.hpp"
#include "op_function/relu.hpp"
#include "op_function/softmax.hpp"
#include "op_function/softmax_cross_entropy.hpp"
#include "op_function/clip.hpp"

#include <memory>
#include <stack>

namespace md {
	extern std::stack<std::shared_ptr<Function>> op_stack;
	extern int start_cnt;
	extern int var_id;

	template<typename F, typename... Ts>
	static inline spvar& compute(Ts&... args) {
		op_stack.push(std::make_shared<F>());
		return op_stack.top()->call(args...)[0];
	}

	class op_func {
	public:
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

		static inline spvar& neg(const spvar& x) {
			return compute<Neg>(x);
		}

		template<typename T0, typename T1>
		static inline spvar& div(T0& x0, T1& x1) {
			return compute<Div>(x0, x1);
		}
	};

	class math {
	public:
		static inline spvar& pow(const spvar& x, const float c) {
			op_stack.push(std::make_shared<Pow>(c));
			return op_stack.top()->call(x)[0];
		}

		static inline spvar& sin(const spvar& x) {
			return compute<Sin>(x);
		}

		static inline spvar& cos(const spvar& x) {
			return compute<Cos>(x);
		}

		static inline spvar& tanh(const spvar& x) {
			return compute<Tanh>(x);
		}

		static inline spvar& exp(const spvar& x) {
			return compute<Exp>(x);
		}
	};

	class inner_util_func {
	private:
		static inline spvar& vreshape(const spvar& x, const xarr_size& target_size) {
			op_stack.push(std::make_shared<Reshape>(target_size));
			return op_stack.top()->call(x)[0];
		}

		static inline spvar& vtranspose(const spvar& x) {
			return compute<Transpose>(x);
		}

		static inline spvar& vdot(const spvar& x, const spvar& W) {
			return compute<MatMul>(x, W);
		}

		static inline spvar& vget_item(const spvar& x, const vec_xslice& slices) {
			op_stack.push(std::make_shared<GetItem>(slices));
			return op_stack.top()->call(x)[0];
		}

		static inline spvar& vclip(const spvar& x, const float x_min, const float x_max) {
			op_stack.push(std::make_shared<Clip>(x_min, x_max));
			return op_stack.top()->call(x)[0];
		}

		friend class spvar;
	};

	class util_func {
	public:
		static spvar& broadcast_to(spvar& x, const xarr_size& shape);
		static spvar& sum_to(spvar& x, const xarr_size& shape);
		static spvar accuracy(const spvar& y, const spvar& t);

		static inline spvar& sum(const spvar& x, const xarr_size& axis, bool keepdims = false) {
			op_stack.push(std::make_shared<Sum>(axis, keepdims));
			return op_stack.top()->call(x)[0];
		}

		static inline spvar& mean_squared_error(const spvar& x0, const spvar& x1) {
			return compute<MeanSquaredError>(x0, x1);
		}

		static inline spvar& linear(const spvar& x, const spvar& W) {
			return compute<Linear>(x, W);
		}

		static inline spvar& linear(const spvar& x, const spvar& W, const spvar& b) {
			return compute<Linear>(x, W, b);
		}

		static inline spvar& sigmoid(const spvar& x) {
			return compute<Sigmoid>(x);
		}

		static inline spvar& softmax(const spvar& x) {
			return softmax(x, { 1 });
		}

		static inline spvar& softmax(const spvar& x, const xarr_size& axis) {
			op_stack.push(std::make_shared<Softmax>(axis));
			return op_stack.top()->call(x)[0];
		}

		static inline spvar& relu(const spvar& x) {
			return compute<ReLU>(x);
		}

		static inline spvar& softmax_cross_entropy(const spvar& x, const spvar& t) {
			return compute<SoftmaxCrossEntropy>(x, t);
		}
	};

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
	}
}
#endif
