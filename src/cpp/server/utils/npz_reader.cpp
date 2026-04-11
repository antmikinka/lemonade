#include "lemon/utils/npz_reader.h"
#include <algorithm>
#include <cctype>

#ifdef _WIN32
#include <cstdlib>
#else
#include <unistd.h>
#endif

namespace lemon {
namespace utils {

NpzReader::NpzReader(const std::string& filepath) : filepath_(filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open NPZ file: " + filepath);
    }

    // Find all .npy files in the ZIP archive
    // Simple ZIP parsing - look for local file headers (PK\x03\x04)
    std::vector<std::pair<std::string, size_t>> entries;

    char buffer[4];
    size_t pos = 0;

    while (file.read(buffer, 4)) {
        pos += 4;

        // Check for local file header signature
        if (buffer[0] == 'P' && buffer[1] == 'K' && buffer[2] == 0x03 && buffer[3] == 0x04) {
            // Read local file header
            uint16_t version, flags, compression;
            uint16_t mod_time, mod_date;
            uint32_t crc32, compressed_size, uncompressed_size;
            uint16_t filename_len, extra_len;

            file.read(reinterpret_cast<char*>(&version), 2);
            file.read(reinterpret_cast<char*>(&flags), 2);
            file.read(reinterpret_cast<char*>(&compression), 2);
            file.read(reinterpret_cast<char*>(&mod_time), 2);
            file.read(reinterpret_cast<char*>(&mod_date), 2);
            file.read(reinterpret_cast<char*>(&crc32), 4);
            file.read(reinterpret_cast<char*>(&compressed_size), 4);
            file.read(reinterpret_cast<char*>(&uncompressed_size), 4);
            file.read(reinterpret_cast<char*>(&filename_len), 2);
            file.read(reinterpret_cast<char*>(&extra_len), 2);

            // Read filename
            std::string filename(filename_len, '\0');
            file.read(&filename[0], filename_len);

            // Skip extra field
            file.seekg(extra_len, std::ios::cur);

            // Check if this is a .npy file
            if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".npy") {
                // Extract base name (remove .npy extension and any path prefix)
                std::string base_name = filename;
                size_t slash_pos = base_name.rfind('/');
                if (slash_pos != std::string::npos) {
                    base_name = base_name.substr(slash_pos + 1);
                }
                if (base_name.size() > 4) {
                    base_name = base_name.substr(0, base_name.size() - 4);
                }

                if (!base_name.empty()) {
                    entries.push_back({base_name, pos});
                    names_.push_back(base_name);
                }
            }

            // Skip to next entry
            file.seekg(compressed_size, std::ios::cur);
            pos += compressed_size;
        } else {
            // Try to find next PK signature
            bool found = false;
            while (file.read(buffer + 3, 1)) {
                pos++;
                if (buffer[3] == 'P') {
                    // Check if we have PK
                    char next[3];
                    if (file.read(next, 3)) {
                        pos += 3;
                        if (next[0] == 'K') {
                            file.seekg(-3, std::ios::cur);
                            pos -= 3;
                            found = true;
                            break;
                        }
                    }
                }
            }
            if (!found) break;
        }
    }

    file.clear();
    file.seekg(0);

    // Parse each .npy file
    for (const auto& entry : entries) {
        ArrayInfo info;
        file.seekg(entry.second);
        read_npy_array(file, entry.first, info);
        arrays_[entry.first] = info;
    }
}

NpzReader::~NpzReader() = default;

std::vector<std::string> NpzReader::get_names() const {
    return names_;
}

bool NpzReader::has_array(const std::string& name) const {
    return arrays_.find(name) != arrays_.end();
}

std::vector<int> NpzReader::get_shape(const std::string& name) const {
    auto it = arrays_.find(name);
    if (it == arrays_.end()) {
        throw std::runtime_error("Array not found: " + name);
    }
    return it->second.shape;
}

std::string NpzReader::get_dtype(const std::string& name) const {
    auto it = arrays_.find(name);
    if (it == arrays_.end()) {
        throw std::runtime_error("Array not found: " + name);
    }
    return it->second.dtype;
}

std::vector<float> NpzReader::get_array(const std::string& name) const {
    auto it = arrays_.find(name);
    if (it == arrays_.end()) {
        throw std::runtime_error("Array not found: " + name);
    }

    const ArrayInfo& info = it->second;

    std::ifstream file(filepath_, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open NPZ file");
    }

    file.seekg(info.offset);

    // Calculate total elements
    size_t total_elements = 1;
    for (int dim : info.shape) {
        total_elements *= dim;
    }

    std::vector<float> result(total_elements);

    if (info.dtype == "<f4" || info.dtype == "float32") {
        // Float32 - direct read
        file.read(reinterpret_cast<char*>(result.data()), total_elements * sizeof(float));
    } else if (info.dtype == "<f8" || info.dtype == "float64") {
        // Float64 - convert to float32
        std::vector<double> temp(total_elements);
        file.read(reinterpret_cast<char*>(temp.data()), total_elements * sizeof(double));
        std::transform(temp.begin(), temp.end(), result.begin(), [](double v) { return static_cast<float>(v); });
    } else {
        throw std::runtime_error("Unsupported dtype: " + info.dtype);
    }

    return result;
}

void NpzReader::read_npy_array(std::istream& zip_stream, const std::string& name, ArrayInfo& info) {
    // Skip compression header (we assume uncompressed for now)
    // Read NPY magic number
    char magic[6];
    zip_stream.read(magic, 6);

    if (magic[0] != '\x93' || magic[1] != 'N' || magic[2] != 'U' ||
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        throw std::runtime_error("Invalid NPY magic number for: " + name);
    }

    // Read version
    uint8_t major_version, minor_version;
    zip_stream.read(reinterpret_cast<char*>(&major_version), 1);
    zip_stream.read(reinterpret_cast<char*>(&minor_version), 1);

    // Read header length
    uint16_t header_len;
    if (major_version == 1) {
        zip_stream.read(reinterpret_cast<char*>(&header_len), 2);
    } else {
        uint32_t header_len32;
        zip_stream.read(reinterpret_cast<char*>(&header_len32), 4);
        header_len = static_cast<uint16_t>(header_len32);
    }

    // Read header string
    std::string header(header_len, '\0');
    zip_stream.read(&header[0], header_len);

    // Parse header (it's a Python dict literal)
    // Example: {'descr': '<f4', 'fortran_order': False, 'shape': (256, 128), }

    // Extract dtype
    size_t descr_pos = header.find("'descr'");
    if (descr_pos == std::string::npos) {
        descr_pos = header.find("\"descr\"");
    }
    if (descr_pos != std::string::npos) {
        size_t colon_pos = header.find(':', descr_pos);
        size_t quote_start = header.find('\'', colon_pos);
        if (quote_start == std::string::npos) {
            quote_start = header.find('"', colon_pos);
        }
        size_t quote_end = header.find_first_of("'\"", quote_start + 1);
        if (quote_start != std::string::npos && quote_end != std::string::npos) {
            info.dtype = header.substr(quote_start + 1, quote_end - quote_start - 1);
        }
    }

    // Extract shape
    size_t shape_pos = header.find("'shape'");
    if (shape_pos == std::string::npos) {
        shape_pos = header.find("\"shape\"");
    }
    if (shape_pos != std::string::npos) {
        size_t paren_start = header.find('(', shape_pos);
        size_t paren_end = header.find(')', paren_start);
        if (paren_start != std::string::npos && paren_end != std::string::npos) {
            std::string shape_str = header.substr(paren_start + 1, paren_end - paren_start - 1);

            // Parse comma-separated integers
            std::stringstream ss(shape_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                // Trim whitespace
                size_t start = token.find_first_not_of(" \t");
                if (start != std::string::npos) {
                    int dim = std::stoi(token.substr(start));
                    info.shape.push_back(dim);
                }
            }
        }
    }

    // Store current position as data offset
    info.offset = static_cast<size_t>(zip_stream.tellg());
    info.size = 0; // Will be calculated from shape and dtype
}

bool NpzReader::find_zip_entry(std::istream& stream, const std::string& name,
                                size_t& data_offset, size_t& compressed_size, size_t& uncompressed_size) {
    // This is a simplified ZIP finder - assumes uncompressed entries
    char buffer[4];
    size_t pos = 0;

    while (stream.read(buffer, 4)) {
        pos += 4;

        if (buffer[0] == 'P' && buffer[1] == 'K' && buffer[2] == 0x03 && buffer[3] == 0x04) {
            // Read local file header
            uint16_t version, flags, compression, filename_len, extra_len;
            uint32_t comp_size, uncomp_size;

            stream.seekg(18, std::ios::cur); // Skip to compressed_size
            stream.read(reinterpret_cast<char*>(&comp_size), 4);
            stream.read(reinterpret_cast<char*>(&uncomp_size), 4);
            stream.read(reinterpret_cast<char*>(&filename_len), 2);
            stream.read(reinterpret_cast<char*>(&extra_len), 2);

            std::string filename(filename_len, '\0');
            stream.read(&filename[0], filename_len);

            if (filename == name + ".npy") {
                data_offset = static_cast<size_t>(stream.tellg());
                compressed_size = comp_size;
                uncompressed_size = uncomp_size;
                return true;
            }

            stream.seekg(extra_len + comp_size, std::ios::cur);
        }
    }

    return false;
}

} // namespace utils
} // namespace lemon
