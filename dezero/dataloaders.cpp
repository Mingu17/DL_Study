#include "dataloaders.hpp"
#include "sp_variable.hpp"
#include "variable.hpp"

namespace md {
	void DataLoader::get(vec_spvar& data_set) {
		int i = iteration;
		xarr_i batch_index = xt::view(index, xt::range(i * batch_size, (i + 1) * batch_size));
		spvar batch_x = spvar::create(dataset.get_train_data(batch_index));
		spvar batch_t = spvar::create(dataset.get_train_label(batch_index));
		data_set.push_back(batch_x);
		data_set.push_back(batch_t);
	}
}