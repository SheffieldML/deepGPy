# deepGPy
Deep GPs with GPy

http://arxiv.org/abs/1412.1370

### Requires
 - GPy (development branch)
 - openMP


### How to run on OS X (10.9+) ###
deepGPy requires OS X does not support openMP by default. To have openMP support on your OS X, you can use a clang-omp lib (i.e., an implementation of the OpenMP C/C++ language extensions in Clang/LLVM compiler). There are two ways:

    a)  Build from source. To download and build your own clang-omp, please refer to: http://clang-omp.github.io/
    
    b)  Using Homebrew version:
        1. brew update
        2. brew install clang-omp
    
After installing clang-omp, copy the omp.h from your clang-omp installation path to your python enviroment. e.g.: 

    cp /usr/local/Cellar/libiomp/20150701/include/libiomp/omp.h /Users/<your user name>/anaconda2/include/.
    
At last, leave 'extra_link_args' and 'libraries' empty in special_einsum.py. e.g.:

    'extra_link_args':[],
    'libraries': []
