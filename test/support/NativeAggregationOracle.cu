/** @file NativeAggregationOracle.cu
 * @brief 使用独立主机循环和稳定排序验证 GPU 算法阶段，不用于生产层次构造。
 */
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include "NativeAggregationOracle.cuh"

namespace thermofem::testoracle {
    /** @brief 内部坐标归并与映射校验设施。 */
    namespace {
        /** @brief 排序前的单条矩阵或图边。@code Entry e{0,1,2}; @endcode */
        struct Entry {
            int row_; ///< 行。
            int column_; ///< 列。
            double value_; ///< 权重或数值。
        };
        /** @brief 校验主机 CSR 边界和有限数值。@param a 输入。@pre 无额外约束。
         * @return 无返回值。@code validate(a); @endcode */
        void validate(const io::HostCsr& a) {
            if (a.rows_ < 1 || a.rows_ != a.columns_ || a.offsets_.size() != static_cast<std::size_t>(a.rows_) + 1 ||
                a.indices_.size() != a.values_.size() ||
                a.values_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                a.offsets_.front() != 0 || a.offsets_.back() != static_cast<int>(a.values_.size())) {
                throw std::invalid_argument("Invalid aggregate setup CSR shape");
            }
            for (int row = 0; row < a.rows_; ++row) {
                if (a.offsets_[row] < 0 || a.offsets_[row] > a.offsets_[row + 1]) {
                    throw std::invalid_argument("Invalid CSR offsets");
                }
            }
            for (std::size_t j = 0; j < a.values_.size(); ++j) {
                if (a.indices_[j] < 0 || a.indices_[j] >= a.columns_ || !std::isfinite(a.values_[j])) {
                    throw std::invalid_argument("Invalid CSR entry");
                }
            }
        }
        /** @brief 将坐标按行列排序并归并。@param n 维度。@param entries 条目。
         * @param maximum 是否使用最大值，否则求和。@pre 合法坐标，数量不超过 INT_MAX。
         * @return 规范化 CSR。@code auto a = compress(n, entries, false); @endcode */
        io::HostCsr compress(int n, std::vector<Entry> entries, bool maximum) {
            if (entries.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::length_error("Aggregate setup index overflow");
            }
            std::stable_sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
                return a.row_ < b.row_ || (a.row_ == b.row_ && a.column_ < b.column_);
            });
            io::HostCsr result;
            result.rows_ = n;
            result.columns_ = n;
            result.offsets_.assign(static_cast<std::size_t>(n) + 1, 0);
            result.indices_.reserve(entries.size());
            result.values_.reserve(entries.size());
            int previousRow = -1, previousColumn = -1;
            for (const auto& entry : entries) {
                if (entry.row_ == previousRow && entry.column_ == previousColumn) {
                    auto& value = result.values_.back();
                    value = maximum ? std::max(value, entry.value_) : value + entry.value_;
                    if (!std::isfinite(value)) {
                        throw std::invalid_argument("Nonfinite aggregate reduction");
                    }
                }
                else {
                    result.indices_.push_back(entry.column_);
                    result.values_.push_back(entry.value_);
                    ++result.offsets_[static_cast<std::size_t>(entry.row_) + 1];
                    previousRow = entry.row_;
                    previousColumn = entry.column_;
                }
            }
            std::partial_sum(result.offsets_.begin(), result.offsets_.end(), result.offsets_.begin());
            return result;
        }
        /** @brief 校验连续映射编号。@param map 映射。@param n 细层维度。
         * @pre 无额外约束。@return 粗层维度。@code auto count = mapSize(map, n); @endcode */
        int mapSize(const std::vector<int>& map, int n) {
            if (map.size() != static_cast<std::size_t>(n) || map.empty()) {
                throw std::invalid_argument("Invalid aggregate map size");
            }
            const int largest = *std::max_element(map.begin(), map.end());
            if (largest < 0 || largest >= n) {
                throw std::invalid_argument("Invalid aggregate map range");
            }
            std::vector<bool> used(static_cast<std::size_t>(largest) + 1, false);
            for (const int value : map) {
                if (value < 0 || value > largest) {
                    throw std::invalid_argument("Invalid aggregate map entry");
                }
                used[value] = true;
            }
            if (std::find(used.begin(), used.end(), false) != used.end()) {
                throw std::invalid_argument("Noncompact aggregate map");
            }
            return largest + 1;
        }
        /** @brief 生成参考算法的对称边优先级。@param i 端点。@param j 端点。
         * @pre 非负编号。@return 确定性哈希。@code auto priority = edgePriority(i,j); @endcode */
        std::uint64_t edgePriority(int i, int j) {
            std::uint64_t h =
                (std::uint64_t(static_cast<unsigned>(std::min(i, j))) << 32) | static_cast<unsigned>(std::max(i, j));
            h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
            h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
            return h ^ (h >> 31);
        }
        /** @brief 构建按编号升序的成员列表，允许吸收轮之间有空聚合。
         * @param map 映射。@param count 粗层编号上界。@pre 映射范围正确。
         * @return 成员表。@code auto groups = members(map, count); @endcode */
        std::vector<std::vector<int>> members(const std::vector<int>& map, int count) {
            std::vector<std::vector<int>> groups(static_cast<std::size_t>(count));
            for (std::size_t i = 0; i < map.size(); ++i) {
                groups[map[i]].push_back(static_cast<int>(i));
            }
            return groups;
        }
    } // namespace

    io::HostCsr canonicalMatrix(const io::HostCsr& matrix) {
        validate(matrix);
        std::vector<Entry> entries;
        entries.reserve(matrix.values_.size());
        for (int row = 0; row < matrix.rows_; ++row) {
            for (int j = matrix.offsets_[row]; j < matrix.offsets_[row + 1]; ++j) {
                entries.push_back({row, matrix.indices_[j], matrix.values_[j]});
            }
        }
        return compress(matrix.rows_, std::move(entries), false);
    }
    StrengthGraph strengthGraph(const io::HostCsr& matrix, double threshold) {
        validate(matrix);
        if (!std::isfinite(threshold) || threshold < 0 || threshold > 1) {
            throw std::invalid_argument("Invalid strength threshold");
        }
        if (matrix.values_.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2)) {
            throw std::length_error("Strength graph index overflow");
        }
        std::vector<Entry> entries;
        entries.reserve(matrix.values_.size() * 2);
        for (int row = 0; row < matrix.rows_; ++row) {
            double maximum{};
            for (int j = matrix.offsets_[row]; j < matrix.offsets_[row + 1]; ++j) {
                if (matrix.indices_[j] != row) {
                    maximum = std::max(maximum, std::abs(matrix.values_[j]));
                }
            }
            if (maximum == 0) {
                continue;
            }
            for (int j = matrix.offsets_[row]; j < matrix.offsets_[row + 1]; ++j) {
                const int column = matrix.indices_[j];
                const double weight = std::abs(matrix.values_[j]) / maximum;
                if (column != row && weight > 0 && weight >= threshold) {
                    entries.push_back({row, column, weight});
                    entries.push_back({column, row, weight});
                }
            }
        }
        return {compress(matrix.rows_, std::move(entries), true)};
    }
    std::vector<int> matchPairs(const StrengthGraph& graph, int rounds) {
        const auto& g = graph.edges_;
        validate(g);
        if (rounds < 1 || rounds > 8) {
            throw std::invalid_argument("Matching rounds must be in [1,8]");
        }
        std::vector<int> mate(g.rows_, -1), proposal(g.rows_, -1);
        for (int round = 0; round < rounds; ++round) {
            for (int i = 0; i < g.rows_; ++i) {
                int best = -1;
                double score{};
                std::uint64_t priority{};
                if (mate[i] < 0) {
                    for (int j = g.offsets_[i]; j < g.offsets_[i + 1]; ++j) {
                        const int neighbor = g.indices_[j];
                        const double weight = g.values_[j];
                        if (neighbor == i || mate[neighbor] >= 0 || weight <= 0 || weight < score) {
                            continue;
                        }
                        const auto candidate = edgePriority(i, neighbor);
                        if (weight > score ||
                            (weight == score &&
                             (best < 0 || candidate > priority || (candidate == priority && neighbor < best)))) {
                            best = neighbor;
                            score = weight;
                            priority = candidate;
                        }
                    }
                }
                proposal[i] = best;
            }
            for (int i = 0; i < g.rows_; ++i) {
                if (proposal[i] >= 0 && proposal[proposal[i]] == i) {
                    mate[i] = proposal[i];
                }
            }
        }
        std::vector<int> prefix(static_cast<std::size_t>(g.rows_) + 1, 0), map(g.rows_);
        for (int i = 0; i < g.rows_; ++i) {
            prefix[i + 1] = prefix[i] + (mate[i] < 0 || i < mate[i] ? 1 : 0);
        }
        for (int i = 0; i < g.rows_; ++i) {
            map[i] = prefix[mate[i] < 0 ? i : std::min(i, mate[i])];
        }
        return map;
    }
    StrengthGraph contractGraph(const StrengthGraph& graph, const std::vector<int>& map) {
        const auto& g = graph.edges_;
        validate(g);
        const int count = mapSize(map, g.rows_);
        std::vector<Entry> entries;
        entries.reserve(g.values_.size());
        for (int row = 0; row < g.rows_; ++row) {
            for (int j = g.offsets_[row]; j < g.offsets_[row + 1]; ++j) {
                if (map[row] != map[g.indices_[j]] && g.values_[j] > 0) {
                    entries.push_back({map[row], map[g.indices_[j]], g.values_[j]});
                }
            }
        }
        return {compress(count, std::move(entries), true)};
    }
    AggregateMap aggregate(const StrengthGraph& graph, int rounds, bool absorbSingletons) {
        const auto& g = graph.edges_;
        auto map = matchPairs(graph, rounds);
        const auto contracted = contractGraph(graph, map);
        const auto second = matchPairs(contracted, rounds);
        for (auto& value : map) {
            value = second[value];
        }
        const int count = mapSize(map, g.rows_);
        if (absorbSingletons) {
            for (int round = 0; round < 2; ++round) {
                const auto groups = members(map, count);
                std::vector<int> proposal(g.rows_, -1);
                std::vector<double> scores(g.rows_, 0);
                for (int i = 0; i < g.rows_; ++i) {
                    if (groups[map[i]].size() != 1) {
                        continue;
                    }
                    for (int j = g.offsets_[i]; j < g.offsets_[i + 1]; ++j) {
                        const int c = map[g.indices_[j]];
                        const double weight = g.values_[j];
                        if (groups[c].size() < 2 || groups[c].size() >= 4 || weight <= 0) {
                            continue;
                        }
                        if (weight > scores[i] || (weight == scores[i] && (proposal[i] < 0 || c < proposal[i]))) {
                            proposal[i] = c;
                            scores[i] = weight;
                        }
                    }
                }
                auto nextMap = map;
                for (int c = 0; c < count; ++c) {
                    if (groups[c].size() < 2 || groups[c].size() >= 4) {
                        continue;
                    }
                    std::vector<int> candidates;
                    for (const int row : groups[c]) {
                        for (int j = g.offsets_[row]; j < g.offsets_[row + 1]; ++j) {
                            const int i = g.indices_[j];
                            if (proposal[i] == c) {
                                candidates.push_back(i);
                            }
                        }
                    }
                    std::sort(candidates.begin(), candidates.end());
                    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
                    std::sort(candidates.begin(), candidates.end(), [&scores](int a, int b) {
                        return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
                    });
                    const auto accepted = std::min(candidates.size(), std::size_t(4) - groups[c].size());
                    for (std::size_t j = 0; j < accepted; ++j) {
                        nextMap[candidates[j]] = c;
                    }
                }
                map.swap(nextMap);
            }
        }
        auto groups = members(map, count);
        std::vector<int> compact(count, -1);
        int used{};
        for (int c = 0; c < count; ++c) {
            if (!groups[c].empty()) {
                compact[c] = used++;
            }
        }
        for (auto& value : map) {
            value = compact[value];
        }
        AggregateMap result;
        result.fineToCoarse_ = std::move(map);
        result.offsets_.push_back(0);
        for (const auto& group : groups) {
            if (group.empty()) {
                continue;
            }
            if (group.size() > 4) {
                throw std::logic_error("Aggregate capacity exceeded");
            }
            ++result.histogram_[group.size() - 1];
            result.members_.insert(result.members_.end(), group.begin(), group.end());
            result.offsets_.push_back(static_cast<int>(result.members_.size()));
        }
        return result;
    }
    io::HostCsr aggregateGalerkin(const io::HostCsr& matrix, const AggregateMap& transfer) {
        validate(matrix);
        const auto& map = transfer.fineToCoarse_;
        const int count = mapSize(map, matrix.rows_);
        std::vector<Entry> entries;
        entries.reserve(matrix.values_.size());
        for (int row = 0; row < matrix.rows_; ++row) {
            for (int j = matrix.offsets_[row]; j < matrix.offsets_[row + 1]; ++j) {
                entries.push_back({map[row], map[matrix.indices_[j]], matrix.values_[j]});
            }
        }
        return compress(count, std::move(entries), false);
    }
} // namespace thermofem::testoracle
