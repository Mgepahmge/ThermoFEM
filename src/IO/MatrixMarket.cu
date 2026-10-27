/** @file MatrixMarket.cu
 * @brief 校验 Matrix Market 文件并将坐标条目整理为主机 CSR。
 */
#include <IO/MatrixMarket.cuh>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace thermofem::io {
    /** @brief 包含 Matrix Market 文本解析的内部辅助操作。 */
    namespace {
        /** @brief 读取下一条非空非注释行。@param input 输入。@param line 输出。
         * @pre 流有效。@return 是否读到有效行。@code while (nextLine(input, line)) {} @endcode */
        bool nextLine(std::istream& input, std::string& line) {
            while (std::getline(input, line)) {
                const auto first = line.find_first_not_of(" \t\r");
                if (first != std::string::npos && line[first] != '%') {
                    return true;
                }
            }
            return false;
        }
        /** @brief 检查一行是否含额外字段。@param row 已解析行。@pre 无额外约束。
         * @return 无返回值。@code requireEnd(row); @endcode */
        void requireEnd(std::istringstream& row) {
            std::string extra;
            if (row >> extra) {
                throw std::runtime_error("Unexpected Matrix Market field");
            }
        }
    } // namespace
    HostCsr readMatrixMarket(std::istream& input) {
        std::string line, banner, object, format, field, symmetry;
        if (!std::getline(input, line)) {
            throw std::runtime_error("Missing Matrix Market header");
        }
        std::istringstream header(line);
        if (!(header >> banner >> object >> format >> field >> symmetry) || banner != "%%MatrixMarket" ||
            object != "matrix" || format != "coordinate" || (field != "real" && field != "integer") ||
            (symmetry != "general" && symmetry != "symmetric")) {
            throw std::runtime_error("Unsupported Matrix Market header");
        }
        requireEnd(header);
        if (!nextLine(input, line)) {
            throw std::runtime_error("Missing Matrix Market dimensions");
        }
        long long rows{}, columns{}, count{};
        std::istringstream dimensions(line);
        constexpr auto limit = std::numeric_limits<int>::max();
        if (!(dimensions >> rows >> columns >> count) || rows < 0 || columns < 0 || count < 0 || rows > limit ||
            columns > limit || count > limit || (symmetry == "symmetric" && rows != columns)) {
            throw std::runtime_error("Invalid Matrix Market dimensions");
        }
        requireEnd(dimensions);
        /** @brief 保存单条坐标，转换期间独占。
         * @code Entry entry{0, 1, 2.0}; @endcode */
        struct Entry {
            int row_; ///< 零基行。
            int column_; ///< 零基列。
            double value_; ///< 数值。
        };
        std::vector<Entry> entries;
        entries.reserve(static_cast<std::size_t>(count));
        int triangle{};
        for (long long k = 0; k < count; ++k) {
            if (!nextLine(input, line)) {
                throw std::runtime_error("Truncated Matrix Market entries");
            }
            std::istringstream record(line);
            long long row{}, column{};
            double value{};
            if (!(record >> row >> column >> value) || row < 1 || row > rows || column < 1 || column > columns ||
                !std::isfinite(value) || (field == "integer" && std::trunc(value) != value)) {
                throw std::runtime_error("Invalid Matrix Market entry");
            }
            requireEnd(record);
            const bool mirror = symmetry == "symmetric" && row != column;
            if (mirror) {
                const int side = row > column ? 1 : -1;
                if (triangle && triangle != side) {
                    throw std::runtime_error("Mixed triangles in symmetric storage");
                }
                triangle = side;
            }
            const std::size_t added = mirror ? 2 : 1;
            if (entries.size() > static_cast<std::size_t>(limit) - added) {
                throw std::runtime_error("Expanded CSR exceeds index range");
            }
            entries.push_back({static_cast<int>(row - 1), static_cast<int>(column - 1), value});
            if (mirror) {
                entries.push_back({static_cast<int>(column - 1), static_cast<int>(row - 1), value});
            }
        }
        if (nextLine(input, line)) {
            throw std::runtime_error("Extra Matrix Market entries");
        }
        if (input.bad()) {
            throw std::runtime_error("Matrix Market input read failure");
        }
        HostCsr result;
        result.rows_ = static_cast<int>(rows);
        result.columns_ = static_cast<int>(columns);
        result.offsets_.assign(static_cast<std::size_t>(rows) + 1, 0);
        result.indices_.resize(entries.size());
        result.values_.resize(entries.size());
        for (const auto& entry : entries) {
            ++result.offsets_[static_cast<std::size_t>(entry.row_) + 1];
        }
        for (std::size_t row = 1; row < result.offsets_.size(); ++row) {
            result.offsets_[row] += result.offsets_[row - 1];
        }
        auto positions = result.offsets_;
        for (const auto& entry : entries) {
            const auto position = static_cast<std::size_t>(positions[entry.row_]++);
            result.indices_[position] = entry.column_;
            result.values_[position] = entry.value_;
        }
        return result;
    }
} // namespace thermofem::io
