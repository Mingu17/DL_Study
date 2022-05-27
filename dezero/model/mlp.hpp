#ifndef __MLP_H__
#define __MLP_H__

#include <functional>
#include "../model.hpp"
#include "../function_set.hpp"
#include "../layer/linear_layer.hpp"

namespace md {
	typedef spvar& (*act_func)(const spvar&);

	class MLP : public Model {
	public:
		MLP(const xarr_size& fc_output_sizes,
			const act_func _activation = util_func::sigmoid) : activation(_activation) {
			init_layers(fc_output_sizes);
		}

		template<std::size_t L>
		MLP(const int(&shape)[L], 
			const act_func _activation = util_func::sigmoid) : activation(_activation) {
			xarr_size l_size(std::begin(shape), std::end(shape));
			init_layers(l_size);
		}

		void init_layers(const xarr_size& output_sizes) {
			for (int i = 0; i < output_sizes.size(); ++i) {
				auto layer = std::make_shared<LinearLayer>(static_cast<int>(output_sizes[i]));
				layers.push_back(layer);
			}
		}

		void forward(vec_spvar& xs) override {
			if (layers.size() > 1) {
				size_t i = 0;
				spvar& x = xs[0];
				for (i = 0; i < layers.size() - 1; ++i) {
					x = activation(layers[i]->call(x)[0]);
				}
				outputs = layers[i]->call(x);
			}
			else if (layers.size() == 1) {
				outputs = layers[0]->call(xs[0]);
			}
			else {
				THROW_EXCEPTION("layer count is zero");
			}
		}
	protected:
		act_func activation;
	};
}
#endif
