#include <iostream>
#include <cstdio>
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
	
	// Get and "display" the output
	auto it = processor.ws.dst;
	for(unsigned int i = 0; i < h ; ++i, std::cout << std::endl)
		for(unsigned int j = 0 ; j < w ; ++j, std::cout << (*it++) << ' ');

}
