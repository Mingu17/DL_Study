#ifndef __LOCAL_EXCEPTION_H__
#define __LOCAL_EXCEPTION_H__

#include <iostream>
#include <string>

namespace md {
	class LocalException {
	public:
		LocalException(std::string comment) {
			err_comment += "[ERROR] ";
			err_comment += comment;
		}

		std::string& get_message() {
			return err_comment;
		}
	protected:
		void print_comment() {
			std::cout << err_comment << std::endl;
		}

	protected:
		std::string err_comment;
	};
}
#endif
