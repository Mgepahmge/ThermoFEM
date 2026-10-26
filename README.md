# ThermoFEM

ThermoFEM 是本科毕业设计“基于有限元的热力学仿真求解器设计与实现”的代码项目，采用 CUDA 和 C++20 开发。

## 项目目标

- 建立稳态与瞬态热传导模型，设计网格、材料属性及边界条件的数据结构。
- 实现单元矩阵计算、全局稀疏矩阵组装，以及温度、热流和对流换热边界条件处理。
- 实现稀疏线性方程组求解与时间步进，完成稳态和瞬态温度场计算。
- 封装求解器接口并导出仿真结果，提供展示模型配置、求解过程和温度场结果的 Demo。
- 通过典型算例与解析解或成熟仿真软件对比，验证计算精度与求解性能。

目前已提供版本接口、主程序及基础数据结构：支持 global、pinned、mapped、managed 策略的 RAII CUDA Buffer，以及向量、行主序稠密矩阵和 CSR 稀疏矩阵。线性代数层使用 cuBLAS/cuSPARSE，提供命名函数与运算符，配有示例与测试。有限元组装与热传导仿真尚未实现。

## 项目结构

```text
ThermoFEM/
├── src/                       # CUDA 实现与主程序，.cu
├── include/thermofem/          # 公共头文件，.cuh
├── include/DataStructure/      # CUDA Buffer、向量、稠密与 CSR 矩阵
├── include/LinearAlgebra/      # cuBLAS/cuSPARSE 运算与运算符
├── test/                      # GoogleTest 测试，.cpp，独立 CMakeLists.txt
├── example/                   # CUDA 示例，独立 CMakeLists.txt
├── scripts/build-docs.sh       # 一键构建文档
├── docs/Doxyfile               # Doxygen 配置
├── cmake/                     # 安装后的 CMake 包配置模板
├── CMakeLists.txt
├── README.md
├── LICENSE
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

在任意目录执行项目脚本即可生成中文 Doxygen 文档：

```bash
bash scripts/build-docs.sh
```

上述相对路径命令从项目根目录执行；从其他目录使用脚本的绝对路径。输出入口为 `build/docs/html/index.html`。如果配置 CMake 时已找到 Doxygen，也可执行 `cmake --build build --target docs`。

## 贡献

本项目用于个人本科毕业设计，原则上不接受任何外部贡献，包括代码、文档或其他形式的提交；请勿提交 Pull Request。

## 许可证

本项目采用 [MIT License](LICENSE)。第三方依赖遵循各自的许可证。
