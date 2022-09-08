# Mmul
简单的矩阵加乘

* **首要适配：CMakeList.txt修改CMAKE_C_COMPILER和CMAKE_CXX_COMPILER即C、C++编译器（位置）**
* * *
* 构建/更新项目:源目录下bash build.sh
* 运行程序:源目录下./build/main
* 修改编译选项:在CMakeList.txt的add_definitions("")里添加
* * *
**How can i use PGO**
1. icpc -g -O2 -vec -xCORE-AVX512 -parallel -par-num-threads=62 -par-schedule-guided=1 -par-affinity=granularity=fine,scatter -prof-gen=threadsafe -prof-dir ./profiled/ main.cpp src/matrix.cpp
3. ./a.out
4. icpc -ipo -g -O2 -vec -xCORE-AVX512 -parallel -par-num-threads=62 -par-schedule-guided=1 -par-affinity=granularity=fine,scatter -prof-use -prof-dir ./profiled/ main.cpp src/matrix.cpp
5. ./a.out
