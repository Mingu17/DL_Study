#ifndef __DATA_LOADERS_H__
#define __DATA_LOADERS_H__

#include "common.hpp"
#include "datasets.hpp"
#include <cmath>

namespace md{
	class DataLoader {
	public:
		DataLoader(const Dataset& _dataset, const int _batch_size, bool _shuffle = true, bool _use_gpu = false)
			:dataset(_dataset), batch_size(_batch_size), shuffle(_shuffle), use_gpu(_use_gpu) {
			data_size = dataset.get_len();
			max_iter = std::ceil(static_cast<float>(data_size) / static_cast<float>(batch_size));
			iteration = 0;
			reset();
		}

		void reset() {
			iteration = 0;
			if (shuffle) {
				index = xt::random::permutation(data_size);
			}
			else {
				index = xt::arange(data_size);
			}
		}

		void operator++(int) {
			next();
		}

		void next() {
			iteration++;
		}

		int operator()() {
			if (iteration >= max_iter) {
				reset();
				return END;
			}
			else return CONTINUE;
		}

		void get(vec_spvar& data_set);

	public:
		const static int CONTINUE = 101;
		const static int END = -1;

	protected:
		const Dataset& dataset;
		int batch_size;
		bool shuffle;
		bool use_gpu;
		size_t data_size;
		int max_iter;
		int iteration;
		xarr_i index;
	};
}

#endif
