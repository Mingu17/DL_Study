#ifndef __MODEL_H__
#define __MODEL_H__

#include "layer.hpp"
#include <memory>
#include <vector>
#include <set>
#include <string>

namespace md {
	typedef std::shared_ptr<Layer> sp_layer;

	class Model : public Layer {
	public:
		Model() {}
		virtual std::set<std::pair<parameter, std::string>>& get_params();

	protected:
		std::vector<sp_layer> layers;
	};
}
#endif
