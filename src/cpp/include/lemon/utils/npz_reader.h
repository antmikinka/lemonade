#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace lemon {
namespace utils {

/**
 * NPZ file reader for loading NumPy compressed arrays.
 * Used for loading voice embeddings from KittenTTS voices.npz files.
 */
class NpzReader {
public:
    explicit NpzReader(const std::string& filepath);
    ~NpzReader();

    // Get list of array names in the NPZ file
    std::vector<std::string> get_names() const;

    // Check if an array exists
    bool has_array(const std::string& name) const;

    // Get array shape
    std::vector<int> get_shape(const std::string& name) const;

    // Get array data type
    std::string get_dtype(const std::string& name) const;

    // Get array data as float vector
    std::vector<float> get_array(const std::string& name) const;

private:
    struct ArrayInfo {
        std::vector<int> shape;
        std::string dtype;
        size_t offset;
        size_t size;
    };

    std::string filepath_;
    std::map<std::string, ArrayInfo> arrays_;
    std::vector<std::string> names_;

    // Read a .npy file from within the zip
    void read_npy_array(std::istream& zip_stream, const std::string& name, ArrayInfo& info);

    // Parse NPY header
    void parse_npy_header(std::istream& stream, ArrayInfo& info);

    // Simple ZIP local header parsing
    bool find_zip_entry(std::istream& stream, const std::string& name, size_t& data_offset, size_t& compressed_size, size_t& uncompressed_size);
};

} // namespace utils
} // namespace lemon
