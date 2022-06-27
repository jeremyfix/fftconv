# C++ 1D/2D convolutions with the Fast Fourier Transform

This repository provides a C++ library for efficiently computing a 1D or 2D convolution using the Fast Fourier Transform implemented by FFTW.

It relies on the fact that the FFT is efficiently computed for specific sizes, namely signal sizes which can be decomposed into a product of the prime factors for which some precomputation have been done by FFTW. At the time of writing, for FFTW, these factors are 13, 11, 7, 5, 3, 2.

In its scope, this single header C++ is restricted to :

- 1D/2D convolutions
- with either zero or constant padding
- for linearly decreasing or gaussian kernels

## How to use

Once compiled and installed, you should be able to compile your code with 

	g++ -o myexample myexample.cpp $(pkg-config --libs --cflags fftconv)

## Example usage

```cpp

#include "fftconv.hpp"

int main(int argc, char* argv[]) {

	unsigned int w, h;
	w = h = 10;
	std::vector<double> data(w*h);
	double sigma(1.0);
	fftconv::Convolution processor(w, h, sigma, data,
			fftconv::KernelType::Gaussian, 
			fftconv::PaddingType::Constant);

	// Fill in some data
	std::fill(data.begin(), data.end(), 1.0);	
	
	// Compute the convolution
	processor.convolve();	

	// Get the result
	// iterate over processor.ws.dst a C++ array of w*h elememnts

	// Fill in some new data and compute the convolution with the same kernel
	std::fill(data.begin(), data.end(), 2.0);
	processor.convolve();
}

```
