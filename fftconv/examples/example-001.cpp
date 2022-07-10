#include <iostream>
#include <cstdio>
#include "fftconv.hpp"


template<typename DATA>
void display(const DATA values, const std::pair<int, int>& shape) {
	// Get and "display" the output
	auto it = values;
	for(unsigned int i = 0; i < shape.first ; ++i, std::cout << std::endl)
		for(unsigned int j = 0 ; j < shape.second ; ++j) {
			std::cout.width(12);
			std::cout.fill(' ');
			std::cout << (*it++) << ' ';	
		}
}

void convolve_and_display(fftconv::Convolution& processor, const std::pair<int, int>& shape) {
	processor.convolve();	
	display(processor.ws.dst, shape);
}

int main(int argc, char* argv[]) {

	unsigned int w, h;
	w = h = 10;
	const std::pair<int, int> shape{h, w};
	std::vector<double> data(w*h);
	double sigma(1.0);
	fftconv::Convolution processor(w, h, sigma, data,
			fftconv::KernelType::Gaussian, 
			fftconv::PaddingType::Constant);

	// Fill in some data
	auto itv  = data.begin();
	for (unsigned int i = 0; i < h ; ++i, itv+=w) {
		std::fill(itv, itv+uint(w/2), 1.0);
		std::fill(itv+uint(w/2), itv+w, 0.0);
	}

	std::cout << "Initial values" << std::endl;
	display(data.data(), shape);

	// Compute the convolution and display the result
	std::cout << "\nConvolution results " << std::endl;
	convolve_and_display(processor, shape);

	// Copy the output back to the input
	std::copy(processor.ws.dst, processor.ws.dst+(w*h), data.begin());
	
	// Convolve it again
	std::cout << "\nConvolution results " << std::endl;
	convolve_and_display(processor, shape);
	
}
