#pragma comment(lib, "libopenblas.lib")

#include <iostream>
#include "dezero/variable.hpp"
#include "dezero/sp_variable.hpp"
#include "dezero/model/mlp.hpp"
#include "dezero/optimizer/adam.hpp"
#include "dezero/dataset/mnist.hpp"
#include "dezero/dataloaders.hpp"

using namespace std;
using namespace md;

bool Common::enable_backprop = true;

int main(int argc, char** argv) {
	try {
		start_op();

		xt::random::seed(0);

		int max_epoch = 5;
		int batch_size = 100;
		int hidden_size = 1000;

		auto train_set = MNIST(true);
		auto test_set = MNIST(false);
		auto train_loader = DataLoader(train_set, batch_size);
		auto test_loader = DataLoader(test_set, batch_size, false);

		auto model = MLP({ hidden_size, hidden_size, 10 }, util_func::relu);
		auto optimizer = Adam();
		optimizer.setup(&model);

		for (int epoch = 0; epoch < max_epoch; ++epoch) {
			double sum_loss = 0.0, sum_acc = 0.0;
			for (; train_loader() != DataLoader::END; train_loader++) {
				vec_spvar train;
				train_loader.get(train);
				auto& y = model(train[0])[0];
				auto& loss = util_func::softmax_cross_entropy(y, train[1]);
				auto acc = util_func::accuracy(y, train[1]);
				model.clear_grad();
				loss->backward();
				optimizer.update();
				sum_loss += loss[0] * train[1]->get_len();
				sum_acc += acc[0] * train[1]->get_len();
				clear_op();
			}
			cout << "epoch: " << epoch + 1 << endl;
			cout << "train loss: " << sum_loss / train_set.get_len() << ", accuracy: " << sum_acc / train_set.get_len() << endl;

			sum_loss = sum_acc = 0.0;

			for (; test_loader() != DataLoader::END; test_loader++) {
				vec_spvar test;
				test_loader.get(test);
				auto& y = model(test[0])[0];
				auto& loss = util_func::softmax_cross_entropy(y, test[1]);
				auto acc = util_func::accuracy(y, test[1]);
				sum_loss += loss[0] * test[1]->get_len();
				sum_acc += acc[0] * test[1]->get_len();
				clear_op();
			}
			cout << "test loss: " << sum_loss / test_set.get_len() << ", accuracy: " << sum_acc / test_set.get_len() << endl;
			clear_op();
		}

		clear_op();

		cout << "End" << endl;
	}
	catch (LocalException& e) {
		cout << e.get_message() << endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}