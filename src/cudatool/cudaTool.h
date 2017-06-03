#pragma once
#ifndef _CUDATOOL_H_
#define _CUDATOOL_H_
#include<vector>
namespace cudaTool {
	bool InitCUDA(void);
	std::vector<std::vector<float>> deelDethDenoise(float* curDepthData, float*& sortedDepthData, int width, int height, int totalCount, float threshold, float percent, bool isLast);
	std::vector<std::vector<float>> deelAllDethDenoise(const std::vector<float>& unSortedDepthData_, int width, int height, int totalCount, float threshold, float percent);	
}
	
	
#endif// _DEPTH_DENOISING_H_

