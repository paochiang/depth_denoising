#pragma once
#ifndef _DEPTH_DENOISING_H_
#define _DEPTH_DENOISING_H_
#include<vector>
#include<opencv2/opencv.hpp>
class DepthDenoising
{
public:
	DepthDenoising(int w, int h);
	~DepthDenoising();
	int width() const { return width_; };
	int height() const { return height_; };
	int& width(){ return width_; };
	int& height(){ return height_; };
	int imageCount()const { return imageCount_; };

	void setCurDepth(const cv::Mat curDepth, const bool isLast);					//mm
	void setCurDepth(const std::vector<unsigned short>curDepth, const bool isLast);	//mm
	void setCurDepth(const std::vector<float>curDepth, const bool isLast);			//m
	void dealCurDepth_GPU();
	void dealCurDepth_CPU();
	void dealAllDepth_GPU();
	void filterDepth(std::vector<float>& depth, const float min_depth = 0.3f, const float max_depth = 4.5f);
	std::vector<std::vector<float>> downloadDenoisedRes();
	cv::Mat getStableResult_mat();				//mm
	std::vector<float> getStableResult_meter(); //m
	std::vector<float> downloadCurDepth();		//m
	cv::Mat toMat_mm(const std::vector<float> depth);	//m -> mm

private:
	const float threshold_ = 0.01f;
	const float percent_ = 0.0f;
	const float min_depth_ = 0.3f; //1.85f; //0.5f;
	const float max_depth_ = 1.0f;// 3.5f;// 1.5f;
	int width_;
	int height_;
	int imageCount_;
	bool isLast_;
	float* depthData_;	//m
	float* sortedDepthData_;	//m
	std::vector<std::vector<float>> depth_res_; //m

	std::vector<float> unSortedDepthData_;	//mm
};
#endif //_DEPTH_DENOISING_H_
