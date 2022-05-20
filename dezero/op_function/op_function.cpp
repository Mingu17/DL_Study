#include "../function_set.hpp"
#include "../variable.hpp"
#include "../sp_variable.hpp"

namespace md {
	/// <summary>
	/// Add class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/
	void Add::forward(const vec_spvar& xs) {
		if (xs.size() != 2) {
			throw LocalException("(Add::forward) - Size mismatch");
		}
		else {
			xarr_f& x0 = xs[0]->get_data();
			xarr_f& x1 = xs[1]->get_data();
			x0_shape = x0.shape();
			x1_shape = x1.shape();
			//xarr_d res = x0 + x1;
			//return vec_spvar({ spvar::create(res) });
			outputs[0] = spvar::create(x0 + x1);
		}
	}

	/// <summary>
	/// Add class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/
	void Add::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Add::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			grads.clear();
			if (x0_shape != x1_shape) {
				//spvar gx0 = util_func::sum_to(gy, x0_shape);// gy.sum_to(x0_shape);
				//spvar gx1 = util_func::sum_to(gy, x1_shape);// gy.sum_to(x1_shape);
				//return vec_spvar({ gx0, gx1 });
				grads.push_back(util_func::sum_to(gy, x0_shape));
				grads.push_back(util_func::sum_to(gy, x1_shape));
			}
			else {
				//return vec_spvar({ gy, gy });
				grads.push_back(gy);
				grads.push_back(gy);
			}
		}
	}

	/// <summary>
	/// BroadcastTo class constructor (Function)
	/// </summary>
	/// <param name="_shape"> - input shape variable</param>
	//BroadcastTo::BroadcastTo(const xarr_size& _shape) 
	//	: shape(_shape) {
	//
	//}

	/// <summary>
	/// BroadcastTo class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/
	void BroadcastTo::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(BroadcastTo::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			x_shape = x.shape();
			//xarr_f res = xt::broadcast(x, shape);
			//return vec_spvar({ spvar::create(res) });
			outputs[0] = spvar::create(xt::broadcast(x, shape));
		}
	}

	/// <summary>
	/// BroadcastTo class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/
	void BroadcastTo::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(BroadcastTo::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			//return vec_spvar({ spvar::create(Utils::sum_to(gy->get_data(), x_shape)) });
			grads.clear();
			grads.push_back(spvar::create(Utils::sum_to(gy->get_data(), x_shape)));
		}
	}


	/// <summary>
	/// Cos class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Cos::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Cos::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			//return vec_spvar({ spvar::create(xt::cos(x)) });
			outputs[0] = spvar::create(xt::cos(x));
		}
	}

	/// <summary>
	/// Cos class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Cos::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Cos::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x = inputs[0];
			//return vec_spvar({ gy * -math::sin(x) });// -sin(x)
			grads.clear();
			grads.push_back(gy * -math::sin(x));
		}
	}

	/// <summary>
	/// Div class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Div::forward(const vec_spvar& xs) {
		if (xs.size() != 2) {
			throw LocalException("(Div::forward) - Size mismatch");
		}
		else {
			xarr_f& x0 = xs[0]->get_data();
			xarr_f& x1 = xs[1]->get_data();
			//xarr_f res = x0 / x1;
			//return vec_spvar({ spvar::create(res) });
			outputs[0] = spvar::create(x0 / x1);
		}
	}

	/// <summary>
	/// Div class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Div::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Div::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x0 = inputs[0];
			spvar& x1 = inputs[1];
			//return vec_spvar({
			//	gy / x1, gy * (-x0 / math::pow(x1, 2)) // x1.pow(2)) //pow(x1, 2))
			//	});
			grads.clear();
			grads.push_back(gy / x1);
			grads.push_back(gy * (-x0 / math::pow(x1, 2)));
		}
	}

	/// <summary>
	/// Exp class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Exp::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Exp::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			//xarr_f res = xt::exp(x);
			//return vec_spvar({ spvar::create(res) });
			outputs[0] = spvar::create(xt::exp(x));
		}
	}

	/// <summary>
	/// Exp class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Exp::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Exp::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& y = outputs[0];
			//return vec_spvar({ gy * y });
			grads.clear();
			grads.push_back(gy * y);
		}
	}

	/// <summary>
	/// Linear class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Linear::forward(const vec_spvar& xs) {
		if (!(xs.size() == 2 || xs.size() == 3)) {
			throw LocalException("(Linear::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			xarr_f& W = xs[1]->get_data();
			xarr_f y = xt::linalg::dot(x, W);

			if (xs.size() == 2) {
				//return vec_spvar({ spvar::create(y) });
				outputs[0] = spvar::create(y);
			}
			else {
				xarr_f& b = xs[2]->get_data();
				//return vec_spvar({ spvar::create(y + b) });
				outputs[0] = spvar::create(y + b);
			}
		}
	}

	/// <summary>
	/// Linear class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Linear::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Linear::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x = inputs[0];
			spvar& W = inputs[1];
			spvar& gx = gy.matmul(W.T());
			spvar& gW = x.T().matmul(gy);

			grads.clear();
			grads.push_back(gx);
			grads.push_back(gW);
			if (inputs.size() > 2) {
				spvar& b = inputs[2];
				grads.push_back(util_func::sum_to(gy, b->get_shape()));
			}
			//if (inputs.size() == 2) {
			//	return vec_spvar({ gx, gW });
			//}
			//else {
			//	spvar& b = inputs[2];
			//	spvar gb = util_func::sum_to(gy, b->get_shape());// gy.sum_to(const_cast<xarr_size&>(b->get_shape()));
			//	return vec_spvar({ gx, gW, gb });
			//}
		}
	}

	/// <summary>
	/// MatMul class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void MatMul::forward(const vec_spvar& xs) {
		if (xs.size() != 2) {
			throw LocalException("(MatMul::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			xarr_f& W = xs[1]->get_data();
			//xarr_f res = xt::linalg::dot(x, W);
			//return vec_spvar({ spvar::create(res) });
			outputs[0] = spvar::create(xt::linalg::dot(x, W));
		}
	}


	/// <summary>
	/// MatMul class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/
	void MatMul::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(MatMul::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x = inputs[0];
			spvar& W = inputs[1];

			spvar gx = gy.dot(W.T());
			spvar gW = x.T().dot(gy);

			//return vec_spvar({ gx, gW });
			grads.clear();
			grads.push_back(gx);
			grads.push_back(gW);
		}
	}

	/// <summary>
	/// MeanSquaredError class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void MeanSquaredError::forward(const vec_spvar& xs) {
		if (xs.size() != 2) {
			throw LocalException("(MeanSquaredError::forward) - Size mismatch");
		}
		else {
			xarr_f& x0 = xs[0]->get_data();
			xarr_f& x1 = xs[1]->get_data();
			xarr_f diff = xt::pow(x0 - x1, 2);
			//xarr_f res = xt::sum(diff) / diff.size();
			//return vec_spvar({ spvar::create(res) });
			outputs[0] = spvar::create(xt::sum(diff) / diff.size());
		}
	}

	/// <summary>
	/// MeanSquaredError class forward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void MeanSquaredError::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(MeanSquaredError::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x0 = inputs[0];
			spvar& x1 = inputs[1];
			spvar& diff = x0 - x1;
			spvar& gx0 = gy * diff * (2.0f / diff->get_size());
			spvar& gx1 = -gx0;
			//return vec_spvar({ gx0, gx1 });
			grads.clear();
			grads.push_back(gx0);
			grads.push_back(gx1);
		}
	}

	/// <summary>
	/// Mul class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Mul::forward(const vec_spvar& xs) {
		if (xs.size() != 2) {
			throw LocalException("(Mul::forward) - Size mismatch");
		}
		else {
			xarr_f& x0 = xs[0]->get_data();
			xarr_f& x1 = xs[1]->get_data();
			//xarr_f res = x0 * x1;
			//return vec_spvar({ spvar::create(res) });
			outputs[0] = spvar::create(x0 * x1);
		}
	}

	/// <summary>
	/// Mul class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Mul::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Mul::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x0 = inputs[0];
			spvar& x1 = inputs[1];
			//return vec_spvar({ gy * x1, gy * x0 });
			grads.clear();
			grads.push_back(gy * x1);
			grads.push_back(gy * x0);
		}
	}

	/// <summary>
	/// Neg class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Neg::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Neg::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			//return vec_spvar({ spvar::create(-x) });
			outputs[0] = spvar::create(-x);
		}
	}

	/// <summary>
	/// Neg class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Neg::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Neg::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			//return vec_spvar({ -gy });
			grads.clear();
			grads.push_back(-gy);
		}
	}

	/// <summary>
	/// Pow class constructor (Function)
	/// </summary>
	/// <param name="_c"> - quotient</param>
	//Pow::Pow(double _c) : c(_c) {
	//
	//}

	/// <summary>
	/// Pow class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Pow::forward(const vec_spvar& xs) {
		//std::cout << "c : " << c << std::endl;
		if (xs.size() != 1) {
			throw LocalException("(Pow::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			//return vec_spvar({ spvar::create(xt::pow(x, c)) });
			outputs[0] = spvar::create(xt::pow(x, c));
		}
	}

	/// <summary>
	/// Pow class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Pow::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Pow::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x = inputs[0];
			//return vec_spvar({ c * math::pow(x, c - 1) * gy });
			grads.clear();
			grads.push_back(c * math::pow(x, c - 1) * gy);
		}
	}

	/// <summary>
	/// Reshape class constructor (Function)
	/// </summary>
	/// <param name="s"> - target shape</param>
	//Reshape::Reshape(const xarr_size& s) 
	//	: shape(s) {
	//
	//}

	/// <summary>
	/// Reshape class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Reshape::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Reshape::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			x_shape = x.shape();
			//xarr_f res = x.reshape(shape);
			//return vec_spvar({ spvar::create(res) });
			outputs[0] = spvar::create(x.reshape(shape));
		}
	}

	/// <summary>
	/// Reshape class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Reshape::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Reshape::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			//return vec_spvar({ reshape(gy, x_shape) });
			//return vec_spvar({ gy.reshape(x_shape) });
			grads.clear();
			grads.push_back(gy.reshape(x_shape));
		}
	}

	/// <summary>
	/// Sigmoid class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Sigmoid::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Sigmoid::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			//xarr_f res = xt::tanh(x * 0.5) * 0.5 + 0.5;
			//return vec_spvar({ spvar::create(res) });
			outputs[0] = spvar::create(xt::tanh(x * 0.5) * 0.5 + 0.5);
		}
	}

	/// <summary>
	/// Sigmoid class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Sigmoid::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Sigmoid::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& y = outputs[0];
			//return vec_spvar({ gy * y * (1.0 - y) });
			grads.clear();
			grads.push_back(gy * y * (1.0 - y));
		}
	}

	/// <summary>
	/// Sin class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Sin::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Sin::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			//return vec_spvar({ spvar::create(xt::sin(x)) });
			outputs[0] = spvar::create(xt::sin(x));
		}
	}

	/// <summary>
	/// Sin class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Sin::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Sin::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x = inputs[0];
			//return vec_spvar({ gy * math::cos(x) }); //cos(x) });
			grads.clear();
			grads.push_back(gy * math::cos(x));
		}
	}

	/// <summary>
	/// Sub class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Sub::forward(const vec_spvar& xs) {
		if (xs.size() != 2) {
			throw LocalException("(Sub::forward) - Size mismatch");
		}
		else {
			xarr_f& x0 = xs[0]->get_data();
			xarr_f& x1 = xs[1]->get_data();
			//return vec_spvar({ spvar::create(x0 - x1) });
			outputs[0] = spvar::create(x0 - x1);
		}
	}


	/// <summary>
	/// Sub class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Sub::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Sub::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			//return vec_spvar({ gy, -gy });
			grads.clear();
			grads.push_back(gy);
			grads.push_back(-gy);
		}
	}

	/// <summary>
	/// Sum class constructor (Function)
	/// </summary>
	/// <param name="_axis"> - standard axis</param>
	/// <param name="_keepdims"> - keep dimensions</param>
	//Sum::Sum(const xarr_size& _axis, bool _keepdims) 
	//	: axis(_axis), keepdims(_keepdims) {
	//
	//}

	/// <summary>
	/// Sum class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Sum::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Sum::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			x_shape = x.shape();
			if (axis.empty()) {
				//return vec_spvar({ spvar::create(xt::sum(x)) });
				outputs[0] = spvar::create(xt::sum(x));
			}
			else {
				//xarr_d res = xt::sum(x, axis);
				//return vec_spvar({ spvar::create(res) });
				outputs[0] = spvar::create(xt::sum(x, axis));
			}
		}
	}

	/// <summary>
	/// Sum class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Sum::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Sum::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& gy_reshape = Utils::reshape_sum_backward(gy, x_shape, axis, keepdims);
			//spvar& gx = broadcast_to(gy_reshape, x_shape);
			//spvar& gx = util_func::broadcast_to(gy_reshape, x_shape);// gy_reshape.broadcast_to(x_shape);
			//return vec_spvar({ gx });
			grads.clear();
			grads.push_back(util_func::broadcast_to(gy_reshape, x_shape));
		}
	}

	/// <summary>
	/// SumTo class constructor
	/// </summary>
	/// <param name="_shape"> - shape of sum?</param>
	//SumTo::SumTo(const xarr_size& _shape) 
	//	: shape(_shape) {
	//
	//}

	/// <summary>
	/// SumTo class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void SumTo::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(SumTo::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			x_shape = x.shape();
			//xarr_f res = Utils::sum_to(x, shape);
			//return vec_spvar({ spvar::create(res) });
			outputs[0] = spvar::create(Utils::sum_to(x, shape));
		}
	}

	/// <summary>
	/// SumTo class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void SumTo::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(SumTo::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			//return vec_spvar({ util_func::broadcast_to(gy, x_shape) });
			grads.clear();
			grads.push_back(util_func::broadcast_to(gy, x_shape));
		}
	}

	/// <summary>
	/// Tanh class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Tanh::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Tanh::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			//return vec_spvar({ spvar::create(xt::tanh(x)) });
			outputs[0] = spvar::create(xt::tanh(x));
		}
	}

	/// <summary>
	/// Tanh class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Tanh::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Tanh::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& y = outputs[0];
			//return vec_spvar({ gy * (1.0 - y * y) }); // vpow(y, 2) ?
			grads.clear();
			grads.push_back(gy * (1.0 - y * y));
		}
	}

	/// <summary>
	/// Transpose class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Transpose::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Transpose::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			//return vec_spvar({ spvar::create(xt::transpose(x)) });
			outputs[0] = spvar::create(xt::transpose(x));
		}
	}

	/// <summary>
	/// Transpose class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Transpose::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Transpose::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			//return vec_spvar({ gy.transpose() });// transpose(gy)
			grads.clear();
			grads.push_back(gy.transpose());
		}
	}

	/// <summary>
	/// ReLU class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void ReLU::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(ReLU::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			//return vec_spvar({ spvar::create(xt::maximum(x, 0.0)) });
			outputs[0] = spvar::create(xt::maximum(x, 0.0));
		}
	}

	/// <summary>
	/// ReLU class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void ReLU::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(ReLU::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x = inputs[0];
			xarr_f mask = x->get_data() > 0.0f;
			//return vec_spvar({ spvar::create(gy->get_data() * mask) });
			grads.clear();
			grads.push_back(spvar::create(gy->get_data() * mask));
		}
	}

	/// <summary>
	/// Softmax class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Softmax::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Softmax::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			xarr_f max = xt::amax(x, axis, xt::keep_dims);
			xarr_f y = x - max;
			xarr_f y_exp = xt::exp(y);
			xarr_f y_exp_sum = xt::sum(y_exp, axis, xt::keep_dims);
			//xarr_f res = y_exp / y_exp_sum;
			//return vec_spvar({ spvar::create(res) });
			outputs[0] = spvar::create(y_exp / y_exp_sum);
		}
	}

	/// <summary>
	/// Softmax class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Softmax::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Softmax::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& y = outputs[0];
			spvar& gx = y * gy;
			spvar& sumdx = util_func::sum(gx, axis, true);// gx.sum(axis, true);
			spvar& y_sumdx = y * sumdx;
			//return vec_spvar({ gx - y_sumdx });
			grads.clear();
			grads.push_back(gx - y_sumdx);
		}
	}

	/// <summary>
	/// GetItem class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void GetItem::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(GetItem::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			//return vec_spvar({ spvar::create(xt::dynamic_view(x, slices)) });
			outputs[0] = spvar::create(xt::dynamic_view(x, slices));
		}
	}

	/// <summary>
	/// GetItem class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void GetItem::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(GetItem::backward) - Size mismatch");
		}
		else {
			spvar& x = inputs[0];
			spvar& gy = gys[0]->get_grad();
			auto f = GetItemGrad(slices, x->get_shape());
			//return f.call(gy);
			grads.clear();
			grads.push_back(f.call(gy)[0]);
		}
	}

	/// <summary>
	/// GetItemGrad class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void GetItemGrad::forward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(GetItemGrad::forward) - Size mismatch");
		}
		else {
			xarr_f& gy = gys[0]->get_data();
			xarr_f gx = xt::zeros<double>(in_shape);
			xarr_f gx_t = xt::dynamic_view(gx, slices);
			gx_t += gy;
			//return vec_spvar({ spvar::create(gx_t) });
			outputs[0] = spvar::create(gx_t);
		}
	}

	/// <summary>
	/// GetItemGrad class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void GetItemGrad::backward(const vec_spvar& ggxs) {
		if (ggxs.size() != 1) {
			throw LocalException("(GetItemGrad::backward) - Size mismatch");
		}
		else {
			spvar& ggx = ggxs[0]->get_grad();
			//return vec_spvar({ ggx.get_item(slices) });
			grads.clear();
			grads.push_back(ggx.get_item(slices));
		}
	}

	/// <summary>
	/// SoftmaxCrossEntropy class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable (last item is t)</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void SoftmaxCrossEntropy::forward(const vec_spvar& xs) {
		if (xs.size() != 2) {
			throw LocalException("(SoftmaxCrossEntropy::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			xarr_f& t = xs[1]->get_data();
			size_t N = x.shape()[0];
			xarr_f log_z = Utils::logsumexp(x, 1);
			xarr_f log_p = x - log_z;
			xt::xarray<size_t> range = xt::arange(N);
			xt::xarray<size_t> ravel = xt::ravel(t);
			using index_type = std::array<std::size_t, 2>;
			std::vector<index_type> indices;
			for (size_t i = 0; i < N; ++i) {
				indices.push_back({ range(i), ravel(i) });
			}
			xarr_f log_p_v = xt::index_view(log_p, indices);
			//xarr_f res = -xt::sum(log_p_v) / static_cast<float>(N);
			//return vec_spvar({ spvar::create(res) });
			outputs[0] = spvar::create(-xt::sum(log_p_v) / static_cast<float>(N));
		}
	}

	/// <summary>
	/// SoftmaxCrossEntropy class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void SoftmaxCrossEntropy::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(SoftmaxCrossEntropy::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			spvar& x = inputs[0];
			spvar& t = inputs[1];
			size_t N = x->get_shape()[0];
			size_t CLS_NUM = x->get_shape()[1];
			gy = gy * (1.0f / static_cast<float>(N));
			spvar& y = util_func::softmax(x);// x.softmax();
			xarr_i t_c = xt::cast<int>(t->get_data());
			xarr_f t_onehot = xt::view(xt::eye(CLS_NUM), xt::keep(t_c), xt::all());
			//spvar& res = (y - spvar::create(t_onehot)) * gy;
			//return vec_spvar({ res });
			grads.clear();
			grads.push_back((y - spvar::create(t_onehot)) * gy);
		}
	}

	/// <summary>
	/// Clip class forward function (Function)
	/// </summary>
	/// <param name="xs"> - input variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Clip::forward(const vec_spvar& xs) {
		if (xs.size() != 1) {
			throw LocalException("(Clip::forward) - Size mismatch");
		}
		else {
			xarr_f& x = xs[0]->get_data();
			//xarr_f y = xt::clip(x, x_min, x_max);
			//return vec_spvar({ spvar::create(y) });
			outputs[0] = spvar::create(xt::clip(x, x_min, x_max));
		}
	}

	/// <summary>
	/// Clip class backward function (Function)
	/// </summary>
	/// <param name="gys"> - gradient variable</param>
	/// <returns></returns>
	/*vec_spvar*/ 
	void Clip::backward(const vec_spvar& gys) {
		if (gys.size() != 1) {
			throw LocalException("(Clip::backward) - Size mismatch");
		}
		else {
			spvar& gy = gys[0]->get_grad();
			xarr_f& x = inputs[0]->get_data();
			xarr_f mask = (x >= x_min) * (x <= x_max);
			//xarr_f gx = gy->get_data() * mask;
			//return vec_spvar({ spvar::create(gx) });
			grads.clear();
			grads.push_back(spvar::create(gy->get_data() * mask));
		}
	}
}