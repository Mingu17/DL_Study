#pragma comment(lib, "libopenblas.lib")

#include <iostream>
#include "dezero/variable.hpp"
#include "dezero/function_set.hpp"
#include "dezero/sp_variable.hpp"
#include "dezero/layer/linear_layer.hpp"
#include "dezero/model/mlp.hpp"
#include "dezero/optimizer/sgd.hpp"
#include "dezero/optimizer/momentum_sgd.hpp"
#include "dezero/optimizer/adagrad.hpp"
#include "dezero/optimizer/adadelta.hpp"
#include "dezero/optimizer/adam.hpp"
//#include <vector>
//#include <memory>
//#include <stack>
//#include <list>
using namespace std;
using namespace md;

bool Common::enable_backprop = true;

//vector<std::reference_wrapper<Variable>> vref;
//list<Variable> vori;
//	
//void push(Variable& vv) {
//	vori.push_back(Variable());
//	vref.push_back(vv);
//	cout << vv << endl;
//}
//
//void push(double d) {
//	auto vv = Variable(d);
//	vori.push_back(vv);
//	vref.push_back(std::reference_wrapper<Variable>(vori.back()));
//	cout << vv << endl;
//}
//
//template<typename T, typename... Ts>
//void push(T& arg, Ts... args) {
//	push(arg);
//	push(args...);
//}

//Variable& sphere(Variable& x, Variable& y) {
//	auto& z = x.pow(2.0) + y.pow(2.0);
//	return z;
//}
//
//Variable& metyas(Variable& x, Variable& y) {
//	auto& z = 0.26 * (x.pow(2.0) + y.pow(2.0)) - 0.48 * x * y;
//	return z;
//}
//
//Variable& goldstein(Variable& x, Variable& y) {
//	auto& z = (1 + (x + y + 1).pow(2.0) * (19 - 14 * x + 3 * x.pow(2.0) - 14 * y + 6 * x * y + 3 * y.pow(2.0))) * 
//		(30 + (2 * x - 3 * y).pow(2.0) * (18 - 32 * x + 12 * x.pow(2.0) + 48 * y - 36 * x * y + 27 * y.pow(2.0)));
//	return z;
//}
//
//double factorial(double n) {
//	if (n > 1) return n * factorial(n - 1);
//	else return 1;
//}
//
//Variable* my_sin(Variable& x, double threshold = 1e-5) {
//	Variable* ty = nullptr;
//	double y = 0;
//
//	for (int i = 0; i < 100000; ++i) {
//		auto c = std::pow(-1, i) / factorial(2.0 * i + 1.0);
//		auto& t = c * x.pow(2.0 * i + 1.0);
//		ty = (ty == nullptr) ? &(y + t) : &(*ty + t);
//		if (std::abs(t.get_data()(0, 0)) < threshold)
//			break;
//	}
//
//	return ty;
//}
//
//Variable& rosenbrock(Variable& x0, Variable& x1) {
//	auto& ret = 100 * (x1 - x0.pow(2.0)).pow(2.0) + (x0 + 1).pow(2.0);
//	return ret;
//}
//
//Variable& f(Variable& x) {
//	auto& y = x.pow(4.0) - 2 * x.pow(2.0);
//	return y;
//}
//
//xarr_d gx2(xarr_d& x) {
//	xarr_d res = 12.0 * xt::pow(x, 2.0) - 4;
//	return res;
//}

spvar& f(spvar& x) {
	//spvar& ret = pow(x, 4) - (2.0 * pow(x, 2));
	spvar& ret = x.pow(4) - (2.0 * x.pow(2));
	return ret;
}

int main(int argc, char** argv) {
	try {
		start_op();
		//auto x = Variable(1.0);
		//auto y = Variable(1.0);

		//auto& z = goldstein(x, y);
		//z.backward();

		//cout << z << endl;
		//cout << x.get_grad() << endl;
		//cout << y.get_grad() << endl;

		//auto x = Variable(Constants::PI / 4);
		//Variable& y = *my_sin(x);
		//y.backward();
		//cout << y.get_data() << endl;
		//cout << x.get_grad() << endl;

		//auto x0 = Variable(0.0);
		//auto x1 = Variable(2.0);
		//double lr = 0.001;
		//int iters = 1000;

		//for (int i = 0; i < iters; ++i) {
		//	cout << x0 << x1 << endl;
		//	auto& y = rosenbrock(x0, x1);

		//	x0.clear_grad();
		//	x1.clear_grad();
		//	y.backward();

		//	x0.get_data() -= lr * x0.get_grad_data();
		//	x1.get_data() -= lr * x1.get_grad_data();
		//}

		//auto x = Variable(2.0);
		//int iters = 10;

		//for (int i = 0; i < iters; ++i) {
		//	cout << i << ", " << x << endl;
		//	auto& y = f(x);
		//	x.clear_grad();
		//	y.backward();

		//	x.get_data() -= x.get_grad() / gx2(x.get_data());
		//}

		//auto x = Variable(2.0);
		//int iters = 10;
		//for (int i = 0; i < iters; ++i) {
		//	cout << i << ", " << x << endl;
		//	auto& y = f(x);
		//	x.clear_grad();
		//	y.backward();

		//	auto& gx = x.get_grad();
		//	xarr_d gx_data = x.get_grad_data();
		//	x.clear_grad();
		//	gx.backward();
		//	xarr_d gx2_data = x.get_grad_data();
		//	x.get_data() -= gx_data / gx2_data;
		//}

		//Test a;
		//a.create();

		//auto x0 = std::make_shared<Variable>(2.0);
		//auto x1 = std::make_shared<Variable>(2.0);
		//auto x2 = std::make_shared<Variable>(3.0);
		//
		//auto& y = (x0 * x1) + x2;
		//y.get()->backward();

		//cout << y.get()->get_data() << endl;
		//cout << x0.get()->get_grad() << endl;

		//auto x = spvar_create(2.0); //std::make_shared<Variable>(2.0);
		//int iters = 10;

		//for (int i = 0; i < iters; ++i) {
		//	cout << i << ", " << x << endl;

		//	auto& y = f(x);
		//	x->clear_grad();
		//	y->backward();

		//	auto gx = x->get_grad();
		//	//auto gx_data = gx->get_data();
		//	x->clear_grad();
		//	gx->backward();
		//	auto gx2 = x->get_grad();
		//	//auto gx2_data = gx2->get_data();

		//	//x->get_data() -= gx_data / gx2_data;
		//	x->get_data() -= gx->get_data() / gx2->get_data();
		//}


		//auto x = spvar::spvar_create(1.0);
		////auto x = spvar_create(xt::linspace(-7.0, 7.0, 200));
		////cout << x << endl;
		//auto& y = x.tanh(); // vtanh(x);
		//y->backward();

		//for (int i = 0; i < 3; ++i) {
		//	auto gx = x->get_grad();
		//	x->clear_grad();
		//	gx->backward();
		//	cout << x->get_grad() << endl;
		//}

		//Operator::start_op();
		//auto x0 = spvar_create(1.0);
		//auto x1 = spvar_create(2.0);
		//auto& y = Operator::add(x0, x1);
		//Operator::clear_op();

		//auto x = spvar_create(xarr_d({ {1, 2, 3}, {4, 5, 6} }));
		//auto& y = sum(x, { 0 });
		//cout << y << endl;
		//y->backward();
		//cout << x->get_grad() << endl;

		//auto x0 = spvar_create(xt::random::randn<double>({ 2, 3, 4, 5 }));
		//auto y0 = sum(x0, xarr_size(), true);
		//cout << x0 << endl;
		//cout << y0 << endl;
		//cout << xt::adapt(y0->get_shape()) << endl;

		//auto x1 = xarr_d({ {{0,1},{2,3}}, {{4,5}, {6,7}}, {{8,9},{10,11}} });
		//xarr_size x1_shape = x1.shape();

		//xarr_d y1 = xt::sum(x1, { 0, 2 });
		//x1_shape[0] = 1;
		//x1_shape[2] = 1;

		//xarr_d y1_reshape = y1;
		//y1_reshape = y1_reshape.reshape(x1_shape);

		//cout << y1 << endl;
		//cout << xt::adapt(y1.shape()) << endl;

		//cout << y1_reshape << endl;
		//cout << xt::adapt(y1_reshape.shape()) << endl;
		//auto x1 = spvar_create(xarr_d({ {{0,1},{2,3}}, {{4,5}, {6,7}}, {{8,9},{10,11}} }));
		//cout << x1 << endl;
		//cout << xt::adapt(x1->get_shape()) << endl;

		//auto& y1 = sum(x1, { 0, 1 });
		//cout << y1 << endl;
		//cout << xt::adapt(y1->get_shape()) << endl;

		//y1->backward();
		//cout << x1->get_grad() << endl;

		//auto x0 = spvar::spvar_create(xarr_d({ {1,2},{3,4} }));
		//auto x1 = spvar::spvar_create(xarr_d({ {5,6},{7,8} }));
		//auto y = x0.dot(x1); //dot(x0, x1);
		//cout << y << endl;

		//y->backward();
		////cout << y << endl;
		//cout << x0->get_grad() << endl;
		//cout << x1->get_grad() << endl;

		//double lr = 0.1;
		//int iters = 100;
		//xt::random::seed(0);

		//xarr_d _x = xt::random::rand<double>({ 100, 1 });
		//auto x = spvar::spvar_create(_x);
		//auto y = spvar::spvar_create(5.0 + 2.0 * _x + _x);
		//auto W = spvar::spvar_create(xt::zeros<double>({ 1, 1 }));
		//auto b = spvar::spvar_create(xt::zeros<double>({ 1 }));

		//for (int i = 0; i < iters; ++i) {
		//	auto y_pred = x.dot(W) + b;
		//	auto& loss = y.mean_squared_error(y_pred);

		//	W->clear_grad();
		//	b->clear_grad();
		//	loss->backward();

		//	W->get_data() -= lr * W->get_grad()->get_data();
		//	b->get_data() -= lr * b->get_grad()->get_data();

		//	cout << W << ", " << b << ", " << loss << endl;
		//	clear_op();
		//}

		//xt::random::seed(0);
		//xarr_size xs = { 100, 1 };
		////auto x = spvar::create(xt::random::rand<double>({ 100, 1 }));
		//auto x = spvar::create(Utils::rand(xs));
		////auto y = spvar::create(xt::sin(2.0 * Constants::PI * x->get_data()) 
		////	+ xt::random::rand<double>({ 100, 1 }));
		//auto y = spvar::create(xt::sin(2.0 * Constants::PI * x->get_data()) + Utils::rand(xs));
		//auto I = 1;
		//auto H = 10;
		//auto O = 1;
		//
		//auto W1 = spvar::create(0.01 * Utils::randn({ I, H }));
		//auto b1 = spvar::create(Utils::zeros({ H }));
		//auto W2 = spvar::create(0.01 * Utils::randn({ H, O }));
		//auto b2 = spvar::create(Utils::zeros({ O }));

		//auto predict = [&](spvar& x) -> spvar& {
		//	spvar& y0 = x.linear(W1, b1);
		//	spvar& y1 = y0.sigmoid();
		//	return y1.linear(W2, b2);
		//};

		//double lr = 0.2;
		//int iters = 10000;

		//for (int i = 0; i < iters; ++i) {
		//	auto& y_pred = predict(x);
		//	auto& loss = y.mean_squared_error(y_pred);

		//	W1->clear_grad();
		//	b1->clear_grad();
		//	W2->clear_grad();
		//	b2->clear_grad();
		//	loss->backward();

		//	W1->get_data() -= lr * W1->get_grad()->get_data();
		//	b1->get_data() -= lr * b1->get_grad()->get_data();
		//	W2->get_data() -= lr * W2->get_grad()->get_data();
		//	b2->get_data() -= lr * b2->get_grad()->get_data();

		//	if (i % 1000 == 0) {
		//		cout << loss << endl;
		//	}
		//	clear_op();
		//}
		
		//xt::random::seed(0);
		//auto x = spvar::create(Utils::rand({ 100, 1 }));
		//auto y = spvar::create(xt::sin(2.0 * Constants::PI * x->get_data()) * Utils::rand({ 100, 1 }));

		////LinearLayer l1(10);
		////LinearLayer l2(1);

		//vector<LinearLayer> v_layer{ LinearLayer(10), LinearLayer(1) };

		//auto predict = [&](spvar& x) -> spvar& {
		//	vec_spvar& y0 = v_layer[0].call(x);
		//	spvar& y1 = y0[0].sigmoid();
		//	return v_layer[1].call(y1)[0];
		//};

		//double lr = 0.2;
		//int iters = 10000;

		//for (int i = 0; i < iters; ++i) {
		//	spvar& y_pred = predict(x);
		//	spvar& loss = y.mean_squared_error(y_pred);
		//	v_layer[0].clear_grad();
		//	v_layer[1].clear_grad();
		//	loss->backward();

		//	for (auto viters = v_layer.begin(); viters != v_layer.end(); viters++) {
		//		auto l1_params = viters->get_params();
		//		for (auto iters = l1_params.begin(); iters != l1_params.end(); iters++) {
		//			parameter p = iters->first;
		//			p->get_data() -= lr * p->get_grad()->get_data();
		//		}
		//	}

		//	if (i % 1000 == 0) {
		//		cout << loss << endl;
		//	}
		//	clear_op();
		//}


		xt::random::seed(0);
		////auto rd = Utils::rand({ 100, 1 });
		//auto x = spvar::create(Utils::rand({ 100, 1 }));
		//auto y = spvar::create(xt::sin(2.0 * Constants::PI * x->get_data()) * Utils::rand({ 100, 1 }));

		//double lr = 0.2;
		//int max_iters = 10000;
		//int hidden_size = 10;

		//auto model = MLP({ hidden_size, 1 });
		//auto optimizer = Adam();
		//optimizer.setup(&model);

		//for (int i = 0; i < max_iters; ++i) {
		//	auto& y_pred = model.call(x);
		//	auto& loss = y.mean_squared_error(y_pred[0]);
		//	model.clear_grad();
		//	loss->backward();

		//	//auto params = model.get_params();
		//	//for (auto p_iter = params.begin(); p_iter != params.end(); p_iter++) {
		//	//	const parameter& p = p_iter->first;
		//	//	p->get_data() -= lr * p->get_grad()->get_data();
		//	//}
		//	optimizer.update();
		//	if (i % 1000 == 0) {
		//		cout << loss << endl;
		//	}
		//	clear_op();
		//}
		auto model = MLP({ 10, 3 });
		auto x = spvar::create(xarr_d({ {0.2, -0.4},{0.3, 0.5},{1.3, -3.2},{2.1, 0.3} }));
		auto t = spvar::create(xarr_d({ 2, 0, 1, 0 }));
		auto& y = model.call(x);
		auto loss = y[0].softmax_cross_entropy(t);
		loss->backward();
		cout << x.clip(0.1, 1.0) << endl;
		cout << y[0] << endl;
		cout << loss << endl;
		clear_op();

		cout << "End" << endl;
	}
	catch (LocalException& e) {
		cout << e.get_message() << endl;
	}

	return 0;
}