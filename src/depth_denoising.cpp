#include "depth_denoising.h"
#include "cudaTool.h"
DepthDenoising::DepthDenoising(int w, int h) :width_(w), height_(h), imageCount_(0), isLast_(false)
{
	depthData_ = NULL;
	sortedDepthData_ = NULL;
	depthData_ = new float[width_ * height_];
	sortedDepthData_ = new float[width_ * height_];	
}

DepthDenoising::~DepthDenoising()
{
	if (depthData_ != NULL) {
		delete[] depthData_;
		depthData_ = NULL;
	}
	if (sortedDepthData_ != NULL) {
		delete[] sortedDepthData_;
		sortedDepthData_ = NULL;
	}
}

void DepthDenoising::setCurDepth(const cv::Mat curDepth, const bool isLast) {
	CV_Assert(!curDepth.empty() && (curDepth.type() == CV_16UC1));
	//imshow("depth", curDepth);
	//waitKey(0);
	imageCount_++;
	isLast_ = isLast;
	for (int r = 0; r < curDepth.rows; r++) {
		for (int c = 0; c < curDepth.cols; c++)
		{
			unsigned index = r * curDepth.cols + c;
			float realDepth = static_cast<float>(curDepth.at<unsigned short>(r, c)) / 1000.0f;
			if (realDepth >= min_depth_ && realDepth <= max_depth_) {
				depthData_[index] = realDepth;
				unSortedDepthData_.push_back(realDepth);
				if (imageCount_ == 1) {
					sortedDepthData_[index] = depthData_[index];
				}
			}
			else {
				depthData_[index] = 0;
				unSortedDepthData_.push_back(0);
				if (imageCount_ == 1) {
					sortedDepthData_[index] = 0;
				}
			}			
		}
	}
	
}

void DepthDenoising::setCurDepth(const std::vector<float>curDepth, const bool isLast) {
	imageCount_++;
	isLast_ = isLast;
	for (int r = 0; r < height_; r++) {
		for (int c = 0; c < width_; c++)
		{
			unsigned index = r * width_ + c;
			float realDepth = curDepth[index];
			if (realDepth >= min_depth_ && realDepth <= max_depth_) {
				depthData_[index] = realDepth;
				unSortedDepthData_.push_back(realDepth);
				if (imageCount_ == 1) {
					sortedDepthData_[index] = depthData_[index];
				}
			}
			else {
				depthData_[index] = 0;
				unSortedDepthData_.push_back(0);
				if (imageCount_ == 1) {
					sortedDepthData_[index] = 0;
				}
			}


		}
	}
}

void DepthDenoising::setCurDepth(const std::vector<unsigned short>curDepth, const bool isLast) {
	imageCount_++;
	isLast_ = isLast;
	for (int r = 0; r < height_; r++) {
		for (int c = 0; c < width_; c++)
		{
			unsigned index = r * width_ + c;
			float realDepth = curDepth[index]/1000.0f;
			if (realDepth >= min_depth_ && realDepth <= max_depth_) {
				depthData_[index] = realDepth;
				unSortedDepthData_.push_back(realDepth);
				if (imageCount_ == 1) {
					sortedDepthData_[index] = depthData_[index];
				}
			}
			else {
				depthData_[index] = 0;
				unSortedDepthData_.push_back(0);
				if (imageCount_ == 1) {
					sortedDepthData_[index] = 0;
				}
			}


		}
	}
}

void DepthDenoising::dealCurDepth_GPU() {
	if (imageCount_ >= 2)
		depth_res_ = cudaTool::deelDethDenoise(depthData_, sortedDepthData_, width_, height_, imageCount_, threshold_, percent_, isLast_);
	else if (imageCount_ == 1){
		depth_res_.resize(height_ * width_);
		for (int r = 0; r < height_; r++) {
			for (int c = 0; c < width_; c++)
			{
				unsigned index = r * width_ + c;
				depth_res_[index].push_back(depthData_[index]);
			}
		}
	}
}

void DepthDenoising::dealAllDepth_GPU() {
	if (imageCount_ >= 2) 
		depth_res_ = cudaTool::deelAllDethDenoise(unSortedDepthData_, width_, height_, imageCount_, threshold_, percent_);
	else if (imageCount_ == 1){
		depth_res_.resize(height_ * width_);
		for (int r = 0; r < height_; r++) {
			for (int c = 0; c < width_; c++)
			{
				unsigned index = r * width_ + c;
				depth_res_[index].push_back(depthData_[index]);
			}
		}
	}
}


void DepthDenoising::dealCurDepth_CPU() {
	if (imageCount_ > 1)
	{
		float* sortedDepthData_t = new float[width_ * height_ * imageCount_];
		for (int r = 0; r < height_; r++) {
			for (int c = 0; c < width_; c++) {
				unsigned index = r * width_ + c;
				int i;
				for (i = imageCount_ - 2; i >= 0; i--) {
					if (sortedDepthData_[index + i * width_ * height_] > depthData_[index]) {
						sortedDepthData_t[index + width_ * height_ * (i + 1)] = sortedDepthData_[index + i * width_ * height_];
					}
					else {
						break;
					}
				}
				sortedDepthData_t[index + width_ * height_ * (i + 1)] = depthData_[index];
				for (int j = i; j >= 0; j--)
					sortedDepthData_t[index + width_ * height_ * j] = sortedDepthData_[index + width_ * height_ * j];
			}
		}

		if (sortedDepthData_) {
			delete[] sortedDepthData_;
			sortedDepthData_ = NULL;
		}
		sortedDepthData_ = sortedDepthData_t;

		if (isLast_) 
		{
			int offset = imageCount_ * percent_;
			std::cout << "image:" << imageCount_ << ",   used:" << static_cast<int>(imageCount_ - 2 * offset) << std::endl;
			depth_res_.clear();
			depth_res_.resize(width_ * height_);
			for (int r = 0; r < height_; r++) 
			{
				for (int c = 0; c < width_; c++) 
				{
					int offset = imageCount_ * percent_;
					unsigned index = r * width_ + c;
					bool flag = false;
					float ave = sortedDepthData_[index + offset * width_ * height_];

					if (sortedDepthData_[index + (imageCount_ - offset - 1) * width_ * height_] - sortedDepthData_[index + offset * width_ * height_] > threshold_ * 3)
						flag = true;
					for (int i = offset; i < imageCount_ - offset - 1; i++)
					{		
						if (flag)
							break;
						float cur = sortedDepthData_[index + i * width_ * height_];
						float behind = sortedDepthData_[index + (i + 1) * width_ * height_];
						if (behind - cur > threshold_) {
							flag = true;
							break;
						}	
						ave += behind;
					}
					if (flag)
					{
						for (int i = 0; i < imageCount_; i++) {
							depth_res_[index].push_back(sortedDepthData_[index + i * width_ * height_]);
						}
					}
					else {
						ave /= (imageCount_ - 2 * offset);
						depth_res_[index].push_back(ave);
					}
				}
			}
		}
	}
	if (imageCount_ == 1) {
		depth_res_.resize(height_ * width_);
		for (int r = 0; r < height_; r++) {
			for (int c = 0; c < width_; c++)
			{
				unsigned index = r * width_ + c;
				depth_res_[index].push_back(depthData_[index]);
			}
		}
	}
}

void DepthDenoising::filterDepth(std::vector<float>& depth, const float min_depth/* = 0.3f*/, const float max_depth/* = 4.5f*/) {
	if (depth.size() <= 0)
	{
		std::cout << "please check input data!" << std::endl;
		return;
	}		
	for (int i = 0; i < depth.size(); i++)
		if (depth[i] < min_depth || depth[i] > max_depth)
			depth[i] = 0;
}

std::vector<std::vector<float>> DepthDenoising::downloadDenoisedRes(){
	return depth_res_;
}

std::vector<float> DepthDenoising::downloadCurDepth() {
	std::vector<float> curDepth;
	if (depthData_ == NULL)
	{
		std::cout << "no depth data!" << std::endl;
		return curDepth;
	}	
	for (int i = 0; i < height_ * width_; i++)
		curDepth.push_back(depthData_[i]);
	return curDepth;
}

cv::Mat DepthDenoising::toMat_mm(const std::vector<float> depth) {
	if (depth.size() != width_ * height_) {
		std::cout << "please check input data!" << std::endl;
	}
	cv::Mat out_depth(height_, width_, CV_16UC1);
	for (int r = 0; r < height_; r++) {
		for (int c = 0; c < width_; c++) {
			unsigned index = r * width_ + c;
			out_depth.at<unsigned short>(r, c) = static_cast<unsigned short>(depth[index] * 1000);
		}
	}
	return out_depth;
}

cv::Mat DepthDenoising::getStableResult_mat() {
	CV_Assert(depth_res_.size() == width_ * height_);
	cv::Mat out(height_, width_, CV_16UC1);
	for (int r = 0; r < height_; r++) {
		for (int c = 0; c < width_; c++) {
			unsigned index = r * width_ + c;
			if (depth_res_[index].size() == 1) {
				//stable depth value
				out.at<unsigned short>(r, c) = depth_res_[index][0] * 1000;			//mm
			}
			else if (depth_res_[index].size() > 1) {
				//unstable depth value
				out.at<unsigned short>(r, c) = 0;
			}
			else if (depth_res_[index].size() < 1) {
				std::cout << "DepthDenoising::getStableResult_mat:row:" << r << "  ,col:" << c << ", cannot find depth!" << std::endl;
				exit(0);
			}
		}
	}
}
std::vector<float> DepthDenoising::getStableResult_meter() {
	CV_Assert(depth_res_.size() == width_ * height_);
	std::vector<float> outV;
	for (int i = 0; i < depth_res_.size(); i++) {
		if (depth_res_[i].size() == 1) {
			//stable depth value
			outV.push_back(depth_res_[i][0]);			//mm
		}
		else if (depth_res_[i].size() > 1) {
			//unstable depth value
			outV.push_back(0);
		}
		else if (depth_res_[i].size() < 1) {
			std::cout << "DepthDenoising::getStableResult_meter: cannot find depth!" << std::endl;
			exit(0);
		}
	}


}