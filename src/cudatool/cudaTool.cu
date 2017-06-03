#include "cudaTool.h"
#include <iostream>
#include <cstdio>
#include <fstream>
const int BLOCK_X = 16;
const int BLOCK_y = 16;
#define SafeCall(call)                                                         \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
	    {                                                                      \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
	    }                                                                      \
}
bool cudaTool::InitCUDA(void)
{
	int count;	
	cudaGetDeviceCount(&count);
	if (count == 0) 
	{
		printf("There is no device.\n");
		return false;
	}
	int i;
	for (i = 0; i<count; i++)
	{
		cudaDeviceProp prop; 
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
		{
			if (prop.major >= 1)
			{
				break;
			}
		}
	}
	printf("Find %d GPUs\n", count);
	if (i == count)
	{
		printf("There is no device supporting CUDA 1.x\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}
__global__ void sortDepth(float* D_cur_depth, float* D_sortedDepthData, int width, int height, int totalCount) {
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < width && dy < height) {
		unsigned int index = dy * width + dx;
		int i;
		for (i = totalCount - 2; i >= 0; i--) {
			if (D_sortedDepthData[index + width * height * i] > D_cur_depth[index]) {
				D_sortedDepthData[index + width * height * (i + 1)] = D_sortedDepthData[index + width * height * i];
			}
			else {
				break;
			}
		}
		D_sortedDepthData[index + width * height * (i + 1)] = D_cur_depth[index];
	}
}

__global__ void dealSortedDepth(float* D_sortedDepthData, char* D_mask, int width, int height, int totalCount, float threshold, float percent) {
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < width && dy < height) {
		unsigned int index = dy * width + dx;
		int offset = totalCount * percent;
		bool flag = false;
		float ave = 0;
		if (D_sortedDepthData[index + (totalCount - offset - 1) * width * height] - D_sortedDepthData[index + offset * width * height] > threshold * 3)
			flag = true;
		for (int i = offset; i < totalCount - offset - 1; i++) {
			if (flag)
				break;
			float cur = D_sortedDepthData[index + i * width * height];
			float behind = D_sortedDepthData[index + (i + 1) * width * height];
			if (behind - cur > threshold) {
				flag = true;
				break;
			}
			ave += cur;
		}
		if (flag)
			D_mask[index] = 0;
		else {
			ave += D_sortedDepthData[index + (totalCount - offset - 1) * width * height];
			ave /= (totalCount - 2 * offset);
			D_mask[index] = 1;
			D_sortedDepthData[index] = ave;
		}
	}

}

__global__ void dealUnSortedAllDepth(float* D_unSortedDepthData, char* D_mask, int width, int height, int totalCount, float threshold, float percent) {
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < width && dy < height) {
		unsigned int index = dy * width + dx;
		//sort
		for (int i = 0; i < totalCount - 1; i++) {
			float curMin = D_unSortedDepthData[index + i * width * height];
			int mark = i;
			for (int j = i + 1; j < totalCount; j++) {
				if (D_unSortedDepthData[index + j * width * height] < curMin) {
					curMin = D_unSortedDepthData[index + j * width * height];
					mark = j;
				}					
			}
			if (mark != i) {
				float t = D_unSortedDepthData[index + i * width * height];
				D_unSortedDepthData[index + i * width * height] = D_unSortedDepthData[index + mark * width * height];
				D_unSortedDepthData[index + mark * width * height] = t;
			}
		}

		//denoising
		int offset = totalCount * percent;
		bool flag = false;
		float ave = 0;
		if (D_unSortedDepthData[index + (totalCount - offset - 1) * width * height] - D_unSortedDepthData[index + offset * width * height] > threshold * 3)
			flag = true;
		for (int i = offset; i < totalCount - offset - 1; i++) {
			if (flag)
				break;
			float cur = D_unSortedDepthData[index + i * width * height];
			float behind = D_unSortedDepthData[index + (i + 1) * width * height];
			if (behind - cur > threshold) {
				flag = true;
				break;
			}
			ave += cur;
		}
		if (flag)
			D_mask[index] = 0;
		else {
			ave += D_unSortedDepthData[index + (totalCount - offset - 1) * width * height];
			ave /= (totalCount - 2 * offset);
			D_mask[index] = 1;
			D_unSortedDepthData[index] = ave;
		}
	}

}

std::vector<std::vector<float>> cudaTool::deelDethDenoise(float* curDepthData, float*& sortedDepthData, int width, int height, int totalCount, float threshold, float percent, bool isLast) {
	std::vector<std::vector<float>> res(width * height);	
	if (totalCount <= 1) {
		std::cout << "cudaTool:deelDethDenoise:totalCount should start at 2!" << std::endl;
		return res;
	}
	dim3 block(BLOCK_X, BLOCK_y);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	float* D_cur_depth;
	float* D_sortedDepthData;
	SafeCall(cudaMalloc((void **)&D_cur_depth, sizeof(float) * width * height));	
	SafeCall(cudaMemcpy(D_cur_depth, curDepthData, sizeof(float) * width * height, cudaMemcpyHostToDevice));
	SafeCall(cudaMalloc((void **)&D_sortedDepthData, sizeof(float) * width * height * totalCount));
	SafeCall(cudaMemcpy(D_sortedDepthData, sortedDepthData, sizeof(float) * width * height *  (totalCount - 1), cudaMemcpyHostToDevice));
	
	sortDepth <<<grid, block>>>(D_cur_depth, D_sortedDepthData, width, height, totalCount);

	if (isLast) 
	{
		char* D_mask;
		SafeCall(cudaMalloc((void **)&D_mask, sizeof(char) * width * height));
		dealSortedDepth << <grid, block >> >(D_sortedDepthData, D_mask, width, height, totalCount, threshold, percent);
		char* mask = new char[width * height];
		float* denoiseRes = new float[width * height * totalCount];
		SafeCall(cudaMemcpy(denoiseRes, D_sortedDepthData, sizeof(float) * width * height * totalCount, cudaMemcpyDeviceToHost));
		SafeCall(cudaMemcpy(mask, D_mask, sizeof(char) * width * height, cudaMemcpyDeviceToHost));

		for (int r = 0; r < height; r++) 
		{
			for (int c = 0; c < width; c++) 
			{
				unsigned index = r * width + c;
				if (mask[index] == 1) 
				{
					res[index].push_back(denoiseRes[index]);
				}
				else {
					for (int i = 0; i < totalCount; i++) 
					{
						res[index].push_back(denoiseRes[index + i * width * height]);
					}
				}
			}
		}
		if (mask) {
			delete[] mask;
			mask = NULL;
		}
		if (denoiseRes) {
			delete[] denoiseRes;
			denoiseRes = NULL;
		}
		cudaFree(D_mask);
	}
	else {
		if (sortedDepthData) {
			delete[] sortedDepthData;
			sortedDepthData = NULL;
		}
		sortedDepthData = new float[sizeof(float) * width * height * totalCount];
		SafeCall(cudaMemcpy(sortedDepthData, D_sortedDepthData, sizeof(float) * width * height * totalCount, cudaMemcpyDeviceToHost));
	}
	cudaFree(D_cur_depth);
	cudaFree(D_sortedDepthData);
	return res;
}

std::vector<std::vector<float>> cudaTool::deelAllDethDenoise(const std::vector<float>& unSortedDepthData, int width, int height, int totalCount, float threshold, float percent) {
	std::vector<std::vector<float>> res(width * height);
	if (totalCount <= 1) {
		std::cout << "cudaTool:deelDethDenoise:totalCount should start at 2!" << std::endl;
		return res;
	}
	dim3 block(BLOCK_X, BLOCK_y);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
	float* allDepthData = new float[unSortedDepthData.size()];
	for (int i = 0; i < unSortedDepthData.size(); i++)
		allDepthData[i] = unSortedDepthData[i];
	float* D_unSortedDepthData;
	SafeCall(cudaMalloc((void **)&D_unSortedDepthData, sizeof(float) * width * height * totalCount));
	SafeCall(cudaMemcpy(D_unSortedDepthData, allDepthData, sizeof(float) * width * height *  totalCount, cudaMemcpyHostToDevice));
	char* D_mask;
	SafeCall(cudaMalloc((void **)&D_mask, sizeof(char) * width * height));

	dealUnSortedAllDepth << <grid, block >> >(D_unSortedDepthData, D_mask, width, height, totalCount, threshold, percent);

	char* mask = new char[width * height];
	SafeCall(cudaMemcpy(allDepthData, D_unSortedDepthData, sizeof(float) * width * height * totalCount, cudaMemcpyDeviceToHost));
	SafeCall(cudaMemcpy(mask, D_mask, sizeof(char) * width * height, cudaMemcpyDeviceToHost));

	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			unsigned index = r * width + c;
			if (mask[index] == 1)
			{
				res[index].push_back(allDepthData[index]);
			}
			else {
				for (int i = 0; i < totalCount; i++)
				{
					res[index].push_back(allDepthData[index + i * width * height]);
				}
			}
		}
	}
	if (mask) {
		delete[] mask;
		mask = NULL;
	}
	if (allDepthData) {
		delete[] allDepthData;
		allDepthData = NULL;
	}
	cudaFree(D_mask);
	cudaFree(D_unSortedDepthData);
	return res;
}
