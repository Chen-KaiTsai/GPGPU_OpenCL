# GPGPU_OpenCL
Some common GPGPU algorithms with OpenCL on Nvidia GPU.

This repository comes with a HackMD document include what optimization skills are provided to different implemented algorithms. This is used by me as a reference when working on implementing different algorithms on GPU. 

# HackMD Document
https://hackmd.io/@Erebustsai/Byul7e-Up

# Algorithms

* Histogram
* Matrix Multiplication
* Convolution / Sobel
* Prefix Scan
* N-body
* Parallel Sorting Algorithms
* Reduction
* SpMV

## Contains

* For using _Windows with Visual Studio 2022_, `.cl` files and `main_xxxx.cpp` files are provided. The main files should be used as examples.
* `main_xxxx.cpp` contain the main functions and additional functions for different algorithm implementations.
* For using _Linux_, `Makefile` is provided.
* `namespace ocl` provide variety of helper functions that can help writing OpenCL programs.
* `namespace CVHelper` provide functions for reading and writing imgs to and from a `float` format. This can be easily change to different types.

## How to Compile

### Windows
* I use Windows for this project so using Visual Studio 2022 is recommanded. Please remember to install and include OpenCV and OpenCL in your project setting.
* For OpenCL headers, I use headers provided in https://github.com/ProjectPhysX/OpenCL-Wrapper . Please refer to the Github page for better detail.

### Linux
* Each algo is in different dir.
* `make`
