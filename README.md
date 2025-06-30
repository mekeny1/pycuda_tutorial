### .1 mamba 

#### 1)使用mamba 

base环境下执行

```shell
conda install conda-forge::mamba
```

使用mamba创建

```shell
mamba create -n pycuda_tutorial python=3.8
mamba activate pycuda_tutorial
```

ipython库安装

```shell
mamba install conda-forge::ipython
```

### .2 日志输出 

终端信息重定位到log文件，如 

```cmd
python 01.hello-world_gpu.py > 01.hello-world_gpu.log 2>&1
```

### .3 调试与性能分析

#### 1)使用Nsight Systems 

```cmd
nsys profile --stats=true -o matrix_ker_report matrix_ker.exe
```

### .4 Scikit-CUDA(Chapter07)

#### 1)Scikit-CUDA安装 

`pip install scikit-cuda` 下载的有问题，上传的为修改过后的 

解释、手动安装与修改如下 

```shell
git clone git@github.com:lebedov/scikit-cuda.git
```

如果使用的cuda11.x，则需修改源码 **scikit-cuda\skcuda\cusolver.py** 的 **_win32_version_list** 列表（在 CUDA 11 中，实际的 **cusolver** DLL 文件名是 **cusolver64_11.dll**，并非 **cusolver64_110.dll** ）

```python
_win32_version_list = [11, 10, 10, 100, 92, 91, 90, 80, 75, 70]
```

参考

> ***https://github.com/lebedov/scikit-cuda/issues/321#issuecomment-992062496*** 

安装 

```shell
cd scikit-cuda
python setup.py install
```

#### 2)ipython命令行执行 

```cmd
In [4]: run linalg.svd.pca.py
D:\ProgramData\miniconda3\envs\pycuda_tutorial\lib\site-packages\scikit_cuda-0.5.4-py3.8.egg\skcuda\cublas.py:284: UserWarning: creating CUBLAS context to get version number
  warnings.warn('creating CUBLAS context to get version number')

In [5]: print(s**2)
[3.00100469e+05 1.00011305e+05 7.79635855e-04 7.76206609e-04
 7.63176358e-04 7.46360689e-04 7.45690777e-04 7.34826841e-04
 7.31621054e-04 7.20023352e-04]

In [6]: print(u[:,0])
[-7.0710683e-01  7.0710671e-01  8.5272154e-07 -9.6822482e-08
 -3.0118460e-07 -6.8284419e-07  6.7019545e-07  2.5714263e-07
 -2.3135274e-07 -1.7833040e-07]

In [7]: print(v[:,1])
[ 0.01290779  0.0074502   0.01737309 ...  0.00029924 -0.01253529
  0.0125266 ]
```

