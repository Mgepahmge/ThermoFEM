/** @file matrix_market_test.cpp
 * @brief 在无 CUDA 依赖的解析路径验证乱序、重复、对称与损坏输入。
 */
#include <IO/MatrixMarket.cuh>
#include <gtest/gtest.h>
#include <sstream>

TEST(MatrixMarketTest, groupsUnsortedCoordinatesAndPreservesDuplicates) {
    std::istringstream input(
        "%%MatrixMarket matrix coordinate real general\n% comment\n3 3 4\n3 1 2\n1 2 4\n1 2 -1\n3 3 5\n");
    const auto matrix = thermofem::io::readMatrixMarket(input);
    EXPECT_EQ(matrix.offsets_, (std::vector<int>{0, 2, 2, 4}));
    EXPECT_EQ(matrix.indices_, (std::vector<int>{1, 1, 0, 2}));
    EXPECT_EQ(matrix.values_, (std::vector<double>{4, -1, 2, 5}));
}
TEST(MatrixMarketTest, expandsSymmetricTriangleAndAcceptsEmpty) {
    std::istringstream input("%%MatrixMarket matrix coordinate integer symmetric\n2 2 3\n1 1 2\n2 1 -1\n2 2 3\n");
    const auto matrix = thermofem::io::readMatrixMarket(input);
    EXPECT_EQ(matrix.offsets_, (std::vector<int>{0, 2, 4}));
    EXPECT_EQ(matrix.indices_, (std::vector<int>{0, 1, 0, 1}));
    EXPECT_EQ(matrix.values_, (std::vector<double>{2, -1, -1, 3}));
    std::istringstream empty("%%MatrixMarket matrix coordinate real general\n0 0 0\n");
    EXPECT_EQ(thermofem::io::readMatrixMarket(empty).offsets_, (std::vector<int>{0}));
}
TEST(MatrixMarketTest, rejectsMalformedUnsupportedAndNonfiniteInputs) {
    const std::vector<std::string> inputs{"",
                                          "%%MatrixMarket matrix array real general\n2 2\n",
                                          "%%MatrixMarket matrix coordinate complex general\n2 2 0\n",
                                          "%%MatrixMarket matrix coordinate real symmetric\n2 3 0\n",
                                          "%%MatrixMarket matrix coordinate real general\n-1 2 0\n",
                                          "%%MatrixMarket matrix coordinate real general\n2147483648 2 0\n",
                                          "%%MatrixMarket matrix coordinate real general\n2 2 1\n",
                                          "%%MatrixMarket matrix coordinate real general\n2 2 1\n0 1 2\n",
                                          "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 3 2\n",
                                          "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 nan\n",
                                          "%%MatrixMarket matrix coordinate integer general\n2 2 1\n1 1 1.5\n",
                                          "%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 2 extra\n",
                                          "%%MatrixMarket matrix coordinate real general\n2 2 0\n1 1 2\n",
                                          "%%MatrixMarket matrix coordinate real symmetric\n2 2 2\n1 2 1\n2 1 1\n"};
    for (const auto& text : inputs) {
        SCOPED_TRACE(text);
        std::istringstream input(text);
        EXPECT_THROW(thermofem::io::readMatrixMarket(input), std::runtime_error);
    }
}
