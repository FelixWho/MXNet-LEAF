# Installing MXNet From Source Without Root Permission

In this document, I use ```PATH_TO_LIB``` as a placeholder for you to choose where you wish to install these libraries. Note since we are in the scenario where root access is disabled, you should choose a path where you have permissions.

## Installing CMake From Source

```
cd ~
wget https://github.com/Kitware/CMake/releases/download/v3.20.3/cmake-3.20.3.tar.gz
tar xzf cmake-3.20.3.tar.gz
cd cmake-3.20.3
./configure --prefix=PATH_TO_LIB
make
make install
```

Don't forget to add library paths to ```/bashrc```.

```
export PATH="{PATH_TO_LIB}/bin:$PATH"
```

## Installing gcc, gfortran From Source

We will use gcc-8.2 as anything above version 8 will cause errors when building MXNet.

```
cd ~
wget https://ftpmirror.gnu.org/gcc/gcc-8.2.0/gcc-8.2.0.tar.gz
tar xf gcc-8.2.0.tar.gz
cd gcc-8.2.0
contrib/download_prerequisites
cd ~
mkdir build; cd build
../gcc-8.2.0/configure -v --build=x86_64-linux-gnu --host=x86_64-linux-gnu --target=x86_64-linux-gnu --prefix=PATH_TO_LIB --enable-checking=release --enable-languages=c,c++,fortran --disable-multilib
```

Finally, we make and install.

```
make -j 8
make install
```

Don't forget to add library paths to ```.bashrc```.

```
export PATH="{PATH_TO_LIB}/bin:$PATH"
export LD_LIBRARY_PATH="{PATH_TO_LIB}/lib64:$LD_LIBRARY_PATH"
export LDFLAGS="-L{PATH_TO_LIB}/lib64"
```

As a quick notice, if you want to add multiple paths to ```LDFLAGS```, simply put 

```
export LDFLAGS="-L{PATH_A} -L{PATH_B} ..."
```

## Installing OpenBLAS From Source

```
cd ~
git clone https://github.com/xianyi/OpenBLAS
cd OpenBLAS
make FC=gfortran
make PREFIX=PATH_TO_LIB install
```

Now add the following to ```.bashrc```

```
export LD_LIBRARY_PATH="{PATH_TO_LIB}/lib:$LD_LIBRARY_PATH"
export BLAS="{PATH_TO_LIB}/lib/libopenblas.a"
export ATLAS="{PATH_TO_LIB}/lib/libopenblas.a"
export OpenBLAS_DIR="{PATH_TO_LIB}"
export OpenBLAS_HOME="{PATH_TO_LIB}"
```

## Installing LAPACK From Source
The trick to installing LAPACK is that you need to install the shared library (.so files), not the static library (.a) files.

Download a recent LAPACK source from http://www.netlib.org/lapack/. In my case I use lapack-3.8.0. Decompress it.

```
cd lapack-3.8.0
mkdir build; cd build
cmake -DCMAKE_INSTALL_PREFIX=PATH_TO_LIB -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_SHARED_LIBS=ON ..
cmake --build .
```

Now add the following to ```.bashrc```

```
export LDFLAGS="-L{PATH_TO_LIB}/lib"
export LAPACK_DIR="{PATH_TO_LIB}/lib"
```

## Installing OpenCV From Source

```
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
unzip opencv.zip
mv opencv-master opencv
mkdir -p build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=PATH_TO_LIB \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_EXAMPLES=ON ../opencv
make -j 8
make install
```

Add the following to ```.bashrc```

```
export OpenCV_DIR="{PATH_TO_LIB}"
```

## Retrieve the MXNet Source Code and Edit Configurations

Now we get to the fun bit.

```
wget http://www.apache.org/dyn/closer.lua?filename=incubator/mxnet/1.8.0/apache-mxnet-src-1.8.0-incubating.tar.gz&action=download
tar -zxvf apache-mxnet-src-1.8.0-incubating.tar.gz
cd apache-mxnet-src-1.8.0-incubating
```

Now in my case, the reason for building MXNet from source was to enable int64 size tensors (the default is int32). To edit the configs, I did

```
cd apache-mxnet-src-1.8.0-incubating/config
vim linux.cmake
vim linux_gpu.cmake
vim darwin.cmake
```

and edited the configurations.

## Build MXNet

```
cd apache-mxnet-src-1.8.0-incubating
mkdir build; cd build
cmake  -D CMAKE_INSTALL_PREFIX=PATH_TO_LIB -DLAPACK_LIBRARIES=PATH_TO_LAPACK_LIB ..
cmake --build . 
```

where ```PATH_TO_LAPACK_LIB``` is the lib folder containing liblapack.so. Finally, we want to add Python bindings so we can import MXNet into our Python programs.

```
python3 -m pip install --user -e ./python
```

## Possible Issues You Might Run Into While Building MXNet

### lgfortran Not Found

Add the following to ```.bashrc```

```
export LDFLAGS="-L{PATH_TO_libgfortran.so}"
``` 

Make sure the -L precedes the path!

## List of Helpful Links Used to Prepare This Tutorial

- https://linuxhostsupport.com/blog/how-to-install-gcc-on-ubuntu-18-04/
- https://hunseblog.wordpress.com/2014/09/15/installing-numpy-and-openblas/
- https://github.com/JuliaLang/julia/issues/6150
- https://stackoverflow.com/questions/17275348/how-to-specify-new-gcc-path-for-cmake
- https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html
https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/
- https://askubuntu.com/questions/1270161/how-to-build-and-link-blas-and-lapack-libraries-by-hand-for-use-on-cluster
https://github.com/dealii/dealii/issues/9169
- http://theoryno3.blogspot.com/2010/12/compiling-lapack-as-shared-library-in.html
- https://stackoverflow.com/questions/36676449/lapack-blas-openblas-proper-installation-from-source-replace-system-libraries
- http://osdf.github.io/blog/numpyscipy-with-openblas-for-ubuntu-1204.html
- https://mxnet.apache.org/versions/1.8.0/get_started/build_from_source
