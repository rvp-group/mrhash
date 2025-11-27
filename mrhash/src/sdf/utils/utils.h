#include <string>
namespace utils {
  //! gets the file extension (ignoring case)
  inline std::string getFileExtension(const std::string& filename) {
    std::string extension = filename.substr(filename.find_last_of(".") + 1);
    return extension;
  }
} // namespace utils
