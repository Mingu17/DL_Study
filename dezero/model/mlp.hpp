#ifndef __MLP_H__
#define __MLP_H__

#include <functional>
#include "../model.hpp"
#include "../function_set.hpp"
#include "../layer/linear_layer.hpp"

namespace md {
	class MLP : public Model {
	public:
		MLP(const xarr_size& fc_output_sizes,
			std::function<spvar& (spvar&)> _activation = util_func::sigmoid) {
			activation = _activation;
			init_layers(fc_output_sizes);
		}

		template<std::size_t L>
		MLP(const int(&shape)[L],
			std::function<spvar& (spvar&)> _activation = util_func::sigmoid) {
			activation = _activation;
			xarr_size l_size(std::begin(shape), std::end(shape));
			init_layers(l_size);
		}

		void init_layers(const xarr_size& output_sizes) {
			for (int i = 0; i < output_sizes.size(); ++i) {
				auto layer = std::make_shared<LinearLayer>(static_cast<int>(output_sizes[i]));
				layers.push_back(layer);
			}
		}

		vec_spvar forward(vec_spvar& xs) {
			if (layers.size() > 1) {
				int i = 0; 
				spvar& x = xs[0];
				for (i = 0; i < layers.size() - 1; ++i) {
					x = activation(layers[i]->call(x)[0]);
				}
				return vec_spvar({ layers[i]->call(x)[0] });
				//const spvar& in = xs[0];
				//spvar& x = activation(layers[0]->call(in)[0]);
				//for (i = 1; i < layers.size() - 1; ++i) {
				//	x = activation(layers[i]->call(x)[0]);
				//}
				//return vec_spvar({ layers[i]->call(x)[0] });
			}
			else if (layers.size() == 1) {
				return vec_spvar({ layers[0]->call(xs[0])[0] });
			}
			else {
				throw LocalException("(MLP::forward) - layer count is zero");
			}
		}

	protected:
		std::function<spvar& (spvar&)> activation;
	};
}
#endif
