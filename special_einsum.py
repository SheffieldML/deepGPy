# Copyright (c) 2015 James Hensman
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import weave

"""
This file provides a weavified version of the function

  np.einsum('ij,ik,il->jkl', A, A, B)

see also test_special_einsum.py
"""

code = """
int n,m,mm,d;
double tmp;
for(n=0;n<N;n++){
  for(m=0;m<M;m++){
    //compute diag
    tmp = A(n,m)*A(n,m);
    for(d=0;d<D;d++){
      res(m,m,d) += tmp*B(n,d);
    }
    //only compute in lower half
    for(mm=0;mm<m;mm++){
      tmp = A(n,m)*A(n,mm);
      for(d=0;d<D;d++){
        res(m,mm,d) += tmp*B(n,d);
      }
    }
  }
}
//make symmpetrical
for(m=0;m<M;m++){
  for(mm=0;mm<m;mm++){
    for(d=0;d<D;d++){
      res(mm,m,d) = res(m,mm,d);
    }
  }
}
"""

def special_einsum(A,B):
    opts = {'headers'           : ['<omp.h>'],
            'extra_compile_args': ['-fopenmp -O3'],
            'extra_link_args'   : ['-lgomp'],
            'libraries': ['gomp']}
    N, M = A.shape
    N2, D = B.shape
    assert N==N2

    res = np.zeros((M, M, D))
    weave.inline(code, ['N','M','D','res','A','B'], type_converters=weave.converters.blitz, support_code='#include <omp.h>', **opts)
    return res

