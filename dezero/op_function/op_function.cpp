#include "add.hpp"
#include "mul.hpp"
#include "neg.hpp"
#include "sub.hpp"
#include "pow.hpp"
#include "div.hpp"
#include "sin.hpp"
#include "cos.hpp"
#include "tanh.hpp"
#include "exp.hpp"
#include "reshape.hpp"
#include "transpose.hpp"
#include "broadcast_to.hpp"
#include "sum_to.hpp"
#include "sum.hpp"
#include "matmul.hpp"
#include "mean_squared_error.hpp"
#include "linear.hpp"
#include "sigmoid.hpp"

#include "relu.hpp"
#include "get_item.hpp"
#include "get_item_grad.hpp"
#include "softmax.hpp"
#include "softmax_cross_entropy.hpp"
#include "clip.hpp"

#include "../variable.hpp"
#include "../sp_variable.hpp"

namespace md {
	/// <summary>
	/// Add class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Add::forward(vec_spvar& xs) {
		if (xs.size() != 2) {
			throw LocalException("(Add::forward) - Size mismatch");
		}
		else {
			xarr_d& x0 = xs[0]->get_data();
			xarr_d& x1 = xs[1]->get_data();
			x0_shape = x0.shape();
			x1_shape = x1.shape();
			xarr_d res = x0 + x1;
			return vec_spvar({ spvar::create(res) });
		}
	}

	/// <summary>
	/// Add class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Add::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Add::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			if (x0_shape != x1_shape) {
				spvar gx0 = gy.sum_to(x0_shape);
				spvar gx1 = gy.sum_to(x1_shape);
				return vec_spvar({ gx0, gx1 });
			}
			else {
				return vec_spvar({ gy, gy });
			}
		}
	}

	/// <summary>
	/// BroadcastTo class constructor (Function)
	/// </summary>
	/// <param name="_shape"> - input shape variable</param>
	BroadcastTo::BroadcastTo(xarr_size& _shape) : shape(_shape) {
	
	}

	/// <summary>
	/// BroadcastTo class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar BroadcastTo::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(BroadcastTo::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			x_shape = x.shape();
			xarr_d res = xt::broadcast(x, shape);
			return vec_spvar({ spvar::create(res) });
		}
	}

	/// <summary>
	/// BroadcastTo class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar BroadcastTo::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(BroadcastTo::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			return vec_spvar({ spvar::create(Utils::sum_to(gy->get_data(), x_shape)) });
		}
	}


	/// <summary>
	/// Cos class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Cos::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Cos::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			return vec_spvar({ spvar::create(xt::cos(x)) });
		}
	}

	/// <summary>
	/// Cos class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Cos::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Cos::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x = inputs[0];
			return vec_spvar({ gy * -x.sin() });// -sin(x)
		}
	}

	/// <summary>
	/// Div class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Div::forward(vec_spvar& xs) {
		if (xs.size() != 2) {
			throw LocalException("(Div::forward) - Size mismatch");
		}
		else {
			xarr_d& x0 = xs[0]->get_data();
			xarr_d& x1 = xs[1]->get_data();
			xarr_d res = x0 / x1;
			return vec_spvar({ spvar::create(res) });
		}
	}

	/// <summary>
	/// Div class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Div::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Div::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x0 = inputs[0];
			spvar& x1 = inputs[1];
			return vec_spvar({
				gy / x1, gy * (-x0 / x1.pow(2)) //pow(x1, 2))
				});
		}
	}

	/// <summary>
	/// Exp class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Exp::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Exp::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			xarr_d res = xt::exp(x);
			return vec_spvar({ spvar::create(res) });
		}
	}

	/// <summary>
	/// Exp class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Exp::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Exp::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& y = outputs[0];
			return vec_spvar({ gy * y });
		}
	}

	/// <summary>
	/// Linear class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Linear::forward(vec_spvar& xs) {
		if (!(xs.size() == 2 || xs.size() == 3)) {
			throw LocalException("(Linear::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			xarr_d& W = xs[1]->get_data();
			xarr_d y = xt::linalg::dot(x, W);

			if (xs.size() == 2) {
				return vec_spvar({ spvar::create(y) });
			}
			else {
				xarr_d& b = xs[2]->get_data();
				return vec_spvar({ spvar::create(y + b) });
			}
		}
	}

	/// <summary>
	/// Linear class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Linear::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Linear::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x = inputs[0];
			spvar& W = inputs[1];

			spvar gx = gy.matmul(W.T());
			spvar gW = x.T().matmul(gy);

			if (inputs.size() == 2) {
				return vec_spvar({ gx, gW });
			}
			else {
				spvar& b = inputs[2];
				spvar gb = gy.sum_to(const_cast<xarr_size&>(b->get_shape()));
				return vec_spvar({ gx, gW, gb });
			}
		}
	}

	/// <summary>
	/// MatMul class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar MatMul::forward(vec_spvar& xs) {
		if (xs.size() != 2) {
			throw LocalException("(MatMul::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			xarr_d& W = xs[1]->get_data();
			xarr_d res = xt::linalg::dot(x, W);
			return vec_spvar({ spvar::create(res) });
		}
	}


	/// <summary>
	/// MatMul class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar MatMul::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(MatMul::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x = inputs[0];
			spvar& W = inputs[1];

			spvar gx = gy.dot(W.T());
			spvar gW = x.T().dot(gy);

			return vec_spvar({ gx, gW });
		}
	}

	/// <summary>
	/// MeanSquaredError class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar MeanSquaredError::forward(vec_spvar& xs) {
		if (xs.size() != 2) {
			throw LocalException("(MeanSquaredError::forward) - Size mismatch");
		}
		else {
			xarr_d& x0 = xs[0]->get_data();
			xarr_d& x1 = xs[1]->get_data();
			xarr_d diff = xt::pow(x0 - x1, 2);
			xarr_d res = xt::sum(diff) / diff.size();
			return vec_spvar({ spvar::create(res) });
		}
	}

	/// <summary>
	/// MeanSquaredError class forward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar MeanSquaredError::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(MeanSquaredError::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x0 = inputs[0];
			spvar& x1 = inputs[1];
			spvar diff = x0 - x1;
			spvar gx0 = gy * diff * (2.0 / diff->get_size());
			spvar gx1 = -gx0;
			return vec_spvar({ gx0, gx1 });
		}
	}

	/// <summary>
	/// Mul class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Mul::forward(vec_spvar& xs) {
		if (xs.size() != 2) {
			throw LocalException("(Mul::forward) - Size mismatch");
		}
		else {
			xarr_d& x0 = xs[0]->get_data();
			xarr_d& x1 = xs[1]->get_data();
			xarr_d res = x0 * x1;
			return vec_spvar({ spvar::create(res) });
		}
	}

	/// <summary>
	/// Mul class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Mul::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Mul::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x0 = inputs[0];
			spvar& x1 = inputs[1];
			return vec_spvar({ gy * x1, gy * x0 });
		}
	}

	/// <summary>
	/// Neg class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Neg::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Neg::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			return vec_spvar({ spvar::create(-x) });
		}
	}

	/// <summary>
	/// Neg class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Neg::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Neg::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			return vec_spvar({ -gy });
		}
	}

	/// <summary>
	/// Pow class constructor (Function)
	/// </summary>
	/// <param name="_c"> - quotient</param>
	Pow::Pow(double _c) : c(_c) {
	
	}

	/// <summary>
	/// Pow class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Pow::forward(vec_spvar& xs) {
		//std::cout << "c : " << c << std::endl;
		if (xs.size() != 1) {
			throw LocalException("(Pow::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			return vec_spvar({ spvar::create(xt::pow(x, c)) });
		}
	}

	/// <summary>
	/// Pow class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Pow::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Pow::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x = inputs[0];
			return vec_spvar({ c * x.pow(c - 1) * gy });
		}
	}

	/// <summary>
	/// Reshape class constructor (Function)
	/// </summary>
	/// <param name="s"> - target shape</param>
	Reshape::Reshape(xarr_size& s) : shape(s) {
	
	}

	/// <summary>
	/// Reshape class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Reshape::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Reshape::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			x_shape = x.shape();
			xarr_d res = x.reshape(shape);
			return vec_spvar({ spvar::create(res) });
		}
	}

	/// <summary>
	/// Reshape class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Reshape::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Reshape::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			//return vec_spvar({ reshape(gy, x_shape) });
			return vec_spvar({ gy.reshape(x_shape) });
		}
	}

	/// <summary>
	/// Sigmoid class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Sigmoid::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Sigmoid::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			xarr_d res = xt::tanh(x * 0.5) * 0.5 + 0.5;
			return vec_spvar({ spvar::create(res) });
		}
	}

	/// <summary>
	/// Sigmoid class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Sigmoid::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Sigmoid::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& y = outputs[0];
			return vec_spvar({ gy * y * (1.0 - y) });
		}
	}

	/// <summary>
	/// Sin class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Sin::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Sin::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			return vec_spvar({ spvar::create(xt::sin(x)) });
		}
	}

	/// <summary>
	/// Sin class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Sin::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Sin::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x = inputs[0];
			return vec_spvar({ gy * x.cos() }); //cos(x) });
		}
	}

	/// <summary>
	/// Sub class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Sub::forward(vec_spvar& xs) {
		if (xs.size() != 2) {
			throw LocalException("(Sub::forward) - Size mismatch");
		}
		else {
			xarr_d& x0 = xs[0]->get_data();
			xarr_d& x1 = xs[1]->get_data();
			return vec_spvar({ spvar::create(x0 - x1) });
		}
	}


	/// <summary>
	/// Sub class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Sub::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Sub::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			return vec_spvar({ gy, -gy });
		}
	}

	/// <summary>
	/// Sum class constructor (Function)
	/// </summary>
	/// <param name="_axis"> - standard axis</param>
	/// <param name="_keepdims"> - keep dimensions</param>
	Sum::Sum(const xarr_size& _axis, bool _keepdims) : axis(_axis), keepdims(_keepdims) {
	
	}

	/// <summary>
	/// Sum class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Sum::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Sum::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			x_shape = x.shape();
			if (axis.empty()) {
				return vec_spvar({ spvar::create(xt::sum(x)) });
			}
			else {
				xarr_d res = xt::sum(x, axis);
				return vec_spvar({ spvar::create(res) });
			}
		}
	}

	/// <summary>
	/// Sum class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Sum::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Sum::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& gy_reshape = Utils::reshape_sum_backward(gy, x_shape, axis, keepdims);
			//spvar& gx = broadcast_to(gy_reshape, x_shape);
			spvar& gx = gy_reshape.broadcast_to(x_shape);
			return vec_spvar({ gx });
		}
	}

	/// <summary>
	/// SumTo class constructor
	/// </summary>
	/// <param name="_shape"> - shape of sum?</param>
	SumTo::SumTo(xarr_size& _shape) : shape(_shape) {
	
	}

	/// <summary>
	/// SumTo class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar SumTo::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(SumTo::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			x_shape = x.shape();
			xarr_d res = Utils::sum_to(x, shape);
			return vec_spvar({ spvar::create(res) });
		}
	}

	/// <summary>
	/// SumTo class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar SumTo::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(SumTo::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			return vec_spvar({ gy.broadcast_to(x_shape) });
		}
	}

	/// <summary>
	/// Tanh class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Tanh::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Tanh::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			return vec_spvar({ spvar::create(xt::tanh(x)) });
		}
	}

	/// <summary>
	/// Tanh class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Tanh::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Tanh::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& y = outputs[0];
			return vec_spvar({ gy * (1.0 - y * y) }); // vpow(y, 2) ?
		}
	}

	/// <summary>
	/// Transpose class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Transpose::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Transpose::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			return vec_spvar({ spvar::create(xt::transpose(x)) });
		}
	}

	/// <summary>
	/// Transpose class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Transpose::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Transpose::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			return vec_spvar({ gy.transpose() });// transpose(gy)
		}
	}

	/// <summary>
	/// ReLU class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar ReLU::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(ReLU::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			return vec_spvar({ spvar::create(xt::maximum(x, 0.0)) });
		}
	}

	/// <summary>
	/// ReLU class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar ReLU::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(ReLU::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x = inputs[0];
			xarr_d mask = x->get_data() > 0;
			return vec_spvar({ spvar::create(gy->get_data() * mask) });
		}
	}

	/// <summary>
	/// Softmax class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Softmax::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Softmax::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			xarr_d y = x - xt::amax(x, axis, xt::keep_dims);
			xarr_d y_exp = xt::exp(y);
			xarr_d res = y_exp / xt::sum(y_exp, axis, xt::keep_dims);
			return vec_spvar({ spvar::create(res) });
		}
	}

	/// <summary>
	/// Softmax class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Softmax::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Softmax::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& y = outputs[0];
			spvar& gx = y * gy;
			spvar& sumdx = gx.sum(axis, true);
			spvar& y_sumdx = y * sumdx;
			return vec_spvar({ gx - y_sumdx });
		}
	}

	/// <summary>
	/// GetItem class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar GetItem::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(GetItem::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			return vec_spvar({ spvar::create(xt::dynamic_view(x, slices)) });
		}
	}

	/// <summary>
	/// GetItem class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar GetItem::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(GetItem::backward) - Size mismatch");
		}
		else {
			spvar& x = inputs[0];
			spvar& gy = gys[0]->get_grad();
			auto f = GetItemGrad(slices, x->get_shape());
			return f.call(gy);
		}
	}

	/// <summary>
	/// GetItemGrad class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar GetItemGrad::forward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(GetItemGrad::forward) - Size mismatch");
		}
		else {
			xarr_d& gy = gys[0]->get_data();
			xarr_d gx = xt::zeros<double>(in_shape);
			xarr_d gx_t = xt::dynamic_view(gx, slices);
			gx_t += gy;
			return vec_spvar({ spvar::create(gx_t) });
		}
	}

	/// <summary>
	/// GetItemGrad class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar GetItemGrad::backward(vec_spvar& ggxs) {
		if (ggxs.size() != 1) {
			throw LocalException("(GetItemGrad::backward) - Size mismatch");
		}
		else {
			spvar& ggx = ggxs[0]->get_grad();
			return vec_spvar({ ggx.get_item(slices) });
		}
	}

	/// <summary>
	/// SoftmaxCrossEntropy class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable (last item is t)</param>
	/// <returns></returns>
	vec_spvar SoftmaxCrossEntropy::forward(vec_spvar& xs) {
		if (xs.size() != 2) {
			throw LocalException("(SoftmaxCrossEntropy::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			xarr_d& t = xs[1]->get_data();
			size_t N = x.shape()[0];
			xarr_d log_z = Utils::logsumexp(x, 1);
			xarr_d log_p = x - log_z;
			xt::xarray<size_t> range = xt::arange(N);
			xt::xarray<size_t> ravel = xt::ravel(t);
			using index_type = std::array<std::size_t, 2>;
			std::vector<index_type> indices;
			for (int i = 0; i < N; ++i) {
				indices.push_back({ range(i), ravel(i) });
			}
			xarr_d log_p_v = xt::index_view(log_p, indices);
			xarr_d res = -xt::sum(log_p_v) / static_cast<double>(N);
			return vec_spvar({ spvar::create(res) });
		}
	}

	/// <summary>
	/// SoftmaxCrossEntropy class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar SoftmaxCrossEntropy::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(SoftmaxCrossEntropy::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x = inputs[0];
			spvar& t = inputs[1];
			size_t N = x->get_shape()[0];
			size_t CLS_NUM = x->get_shape()[1];
			gy = gy * (1.0 / static_cast<double>(N));
			spvar& y = x.softmax();
			xarr_d t_onehot = xt::view(xt::eye(CLS_NUM), xt::keep(t->get_data()), xt::all());
			spvar& res = (y - spvar::create(t_onehot)) * gy;
			return vec_spvar({ res });
		}
	}

	/// <summary>
	/// Clip class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	vec_spvar Clip::forward(vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Clip::forward) - Size mismatch");
		}
		else {
			xarr_d& x = xs[0]->get_data();
			xarr_d y = xt::clip(x, x_min, x_max);
			return vec_spvar({ spvar::create(y) });
		}
	}

	/// <summary>
	/// Clip class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	vec_spvar Clip::backward(vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Clip::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			xarr_d& x = inputs[0]->get_data();
			xarr_d mask = (x >= x_min) * (x <= x_max);
			xarr_d gx = gy->get_data() * mask;
			return vec_spvar({ spvar::create(gx) });
		}
	}
}