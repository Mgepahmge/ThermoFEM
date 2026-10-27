# ThermoFEM

ThermoFEM 是本科毕业设计“基于有限元的热力学仿真求解器设计与实现”的代码项目，采用 CUDA 和 C++20 开发。

## 项目目标

- 建立稳态与瞬态热传导模型，设计网格、材料属性及边界条件的数据结构。
- 实现单元矩阵计算、全局稀疏矩阵组装，以及温度、热流和对流换热边界条件处理。
- 实现稀疏线性方程组求解与时间步进，完成稳态和瞬态温度场计算。
- 封装求解器接口并导出仿真结果，提供展示模型配置、求解过程和温度场结果的 Demo。
- 通过典型算例与解析解或成熟仿真软件对比，验证计算精度与求解性能。

基础数据结构包括支持 global、pinned、mapped、managed 策略的 RAII CUDA Buffer，以及向量、行主序稠密矩阵和 CSR 稀疏矩阵。线性代数层使用 cuBLAS/cuSPARSE，提供命名函数与运算符。求解流程包括加权 Jacobi 平滑器、GPU 原生聚合 AMG 预条件子和右预条件 BiCGStab 求解器，通过独立基类的 setup/apply 接口组合，不使用 cuSOLVER。AMG 提供 V、W、F 三种循环。有限元组装与热传导仿真不在现有接口的功能范围内。

## 项目结构

```text
ThermoFEM/
├── src/                       # CUDA 实现与主程序，.cu
├── include/thermofem/          # 公共头文件，.cuh
├── include/DataStructure/      # CUDA Buffer、向量、稠密与 CSR 矩阵
├── include/LinearAlgebra/      # cuBLAS/cuSPARSE 运算与运算符
├── include/Multigrid/          # GPU 聚合、传输、粗层 LU 与独立循环调度
├── include/Smoother/           # 平滑器组件
├── include/Preconditioner/     # AMG 与恒等预条件子
├── include/Solver/             # 分层基类、求解报告与 BiCGStab
├── test/                      # GoogleTest 测试，.cpp，独立 CMakeLists.txt
├── example/                   # CUDA 示例，独立 CMakeLists.txt
├── scripts/build-docs.sh       # 一键构建文档
├── scripts/format.sh           # 格式化与格式检查
├── docs/Doxyfile               # Doxygen 配置
├── cmake/                     # 安装后的 CMake 包配置模板
├── CMakeLists.txt
├── README.md
├── LICENSE
├── .clang-format
└── .gitignore
```

除测试用例外，所有源文件使用 `.cu`，所有项目头文件使用 `.cuh`。

## 环境要求

- Linux 与 Bash（文档脚本使用 Bash）。
- CMake 3.24 或更新版本。
- CUDA Toolkit 12.0 或更新版本，以及该版本支持、具备 C++20 支持的宿主 C++ 编译器。
- 运行 GPU 计算需要 CUDA 兼容的 NVIDIA GPU 和与 Toolkit 兼容的驱动。
- GoogleTest：优先使用已安装的 CMake 包；未找到时自动获取 `v1.17.0`，首次配置需要访问 GitHub。可通过 `FETCHCONTENT_SOURCE_DIR_GOOGLETEST` 指定离线源码目录。
- Doxygen（生成文档时需要，不依赖 Graphviz 或 LaTeX）。

默认编译架构为 `75`，请根据目标 GPU 和 Toolkit 支持范围设置 `CMAKE_CUDA_ARCHITECTURES`；例如 Ada GPU 可指定 `89`。`example-1` 仅调用版本接口；`example-2` 包含真实 CUDA 内核，并要求设备支持映射内存和统一内存。测试在设备不可用时明确跳过 GPU 用例。

## 构建与运行

在项目根目录执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/thermofem
./build/example/example-1
./build/example/example-2
./build/example/example-3
./build/example/example-4
ctest --test-dir build --output-on-failure
```

指定 CUDA 编译器和目标架构：

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

默认启用测试和示例；可分别通过 `-DBUILD_TESTING=OFF` 和 `-DTHERMOFEM_BUILD_EXAMPLES=OFF` 关闭。使用本地 GoogleTest 源码：

```bash
cmake -S . -B build \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest
```

## 安装

```bash
cmake --install build --prefix "$PWD/install"
```

安装内容包括主程序、静态库、公共 `.cuh` 头文件、MIT 许可证和 CMake 包配置。示例、测试及其依赖不安装。

其他 CUDA/C++20 CMake 项目可通过安装前缀查找并链接：

```cmake
find_package(ThermoFEM 0.1 CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE ThermoFEM::thermofem)
```

配置使用方时传入 `-DCMAKE_PREFIX_PATH=/path/to/ThermoFEM/install`，并确保 CUDA Toolkit 可被找到。

## 文档

基础数据结构的接口、内存策略、复制与移动语义及同步要求见 [基础数据结构](docs/data-structures.md)。
线性代数接口及当前限制见 [线性代数](docs/linear-algebra.md)。
分层接口、求解参数、异常保证及 V/W/F 语义见 [求解组件](docs/solver.md)。
矩阵输入、命令行求解及输出含义见 [Matrix Market 用户手册](docs/matrix-market.md)。
完整文档分类见 [文档目录](docs/index.md)。

在任意目录执行项目脚本即可生成中文 Doxygen 文档：

```bash
bash scripts/build-docs.sh
```

上述相对路径命令从项目根目录执行；从其他目录使用脚本的绝对路径。输出入口为 `build/docs/html/index.html`。如果配置 CMake 时已找到 Doxygen，也可执行 `cmake --build build --target docs`。

## 代码格式

代码格式由根目录 `.clang-format` 统一定义，CUDA 源文件、项目头文件和 C++ 测试均须格式化。
需要支持 `InsertBraces` 与 `InsertNewlineAtEOF` 的 clang-format，建议统一使用 23.1.3。

```bash
bash scripts/format.sh --write
bash scripts/format.sh --check
```

脚本只处理 src、include、test 和 example 目录内的 CUDA 源文件、CUDA 头文件及 C++ 测试文件。
可通过 `CLANG_FORMAT=/path/to/clang-format` 指定程序。配置 CMake 时若找到 clang-format，
还可使用 `cmake --build build --target format` 或 `format-check`。

## 持续集成

GitHub Actions 使用托管的 Ubuntu runner，仅执行代码格式检查与文档构建发布，不配置 CUDA 构建或 GPU 测试。
代码格式检查在 push、Pull Request 或手动触发时执行，固定使用 clang-format 23.1.3。
文档工作流在 main 分支 push 或手动触发时执行，直接调用 Doxygen，无需 CUDA Toolkit。

生成的 HTML 通过 SSH 发布至 [ThermoFEM-Doc](https://github.com/Mgepahmge/ThermoFEM-Doc) 的 gh-pages 分支根目录。
文档未变化时不创建提交；发布保留分支历史，并清除已删除的页面。

首次配置由仓库维护者完成：

1. 在文档仓库的 Settings → Deploy keys 添加专用 SSH 公钥，并启用 Allow write access。
2. 在代码仓库 ThermoFEM 的 Settings → Secrets and variables → Actions 添加名为 `DOCS_DEPLOY_KEY` 的 secret，内容为对应的完整私钥。
3. 首次工作流创建 gh-pages 分支后，在文档仓库 Settings → Pages 选择 Deploy from a branch，分支 gh-pages，目录 / (root)。

文档站点地址为 <https://mgepahmge.github.io/ThermoFEM-Doc/>。
私钥只存放在受保护的凭据文件和 Actions secret 中，不提交到任一仓库。
工作流固定 GitHub 的 SSH 主机公钥并启用严格校验；GitHub 轮换主机公钥时需核对官方信息后更新工作流。

## 贡献

本项目用于个人本科毕业设计，原则上不接受任何外部贡献，包括代码、文档或其他形式的提交；请勿提交 Pull Request。

## 算法参考

Jacobi、AMG 与 BiCGStab 部分的算法组织参考作者 Mgepahmge 的非公开项目 **YHAMG-GPU**。
参考项目不作为公开依赖分发；本项目保持独立的接口、实现与测试。

## 许可证

本项目采用 [MIT License](LICENSE)。第三方依赖遵循各自的许可证。
