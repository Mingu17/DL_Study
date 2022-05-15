#ifndef __LINEAR_LAYER_H__
#define __LINEAR_LAYER_H__

#include "../layer.hpp"
#include "../utils.hpp"

namespace md {
	class LinearLayer : public Layer {
	public:
		LinearLayer(const int _out_size, const bool nobias = false, const int _in_size = 0) {
			in_size = _in_size;
			out_size = _out_size;

			if (in_size != 0) {
				_init_W();
			}

			if (nobias) {
				b = parameter::create(Utils::zeros({ 0 }));
			}
			else {
				b = parameter::create(Utils::zeros({ out_size }));
				b->set_name("b");
				params.insert(std::make_pair(b, "b"));
			}
		}

		vec_spvar forward(vec_spvar& xs) {
			//const spvar& x = xs[0];
			spvar& x = xs[0];
			if (W == nullptr) {
				in_size = x->get_shape()[1];// static_cast<int>(x->get_shape()[1]);
				_init_W();
			}
			//return vec_spvar({ x.linear(W, b) });
			return vec_spvar({ util_func::linear(x, W, b) });
		}

	protected:
		void _init_W() {
			xarr_d w_data = Utils::randn({ in_size, out_size }) * std::sqrt(1.0 / in_size);
			W = parameter::create(w_data);
			W->set_name("W");
			params.insert(std::make_pair(W, "W"));
		}

	protected:
		int in_size;
		int out_size;
		parameter W;
		parameter b;
	};
}

#endif
