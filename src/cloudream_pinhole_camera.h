#ifndef _CLOUDREAM_PINHOLE_CAMERA_H_
#define _CLOUDREAM_PINHOLE_CAMERA_H_

#include <vector>
#include <opencv2/opencv.hpp>


namespace CLOUDREAM {

	class PinholeCameraIntrinsics {
	public:
		PinholeCameraIntrinsics()
			: fx_(-1.0), fy_(-1.0), cx_(0.0), cy_(0.0) {
			memset(distortion_parameters_, 0, sizeof(double) * 5);
			nRows_ = -1;
			nCols_ = -1;
		}
		// copy constructor
		PinholeCameraIntrinsics(const PinholeCameraIntrinsics& K)
		{
			fx_ = K.fx_;
			fy_ = K.fy_;
			cx_ = K.cx_;
			cy_ = K.cy_;
			memcpy(distortion_parameters_, K.distortion_parameters_, sizeof(double) * 5);
			nRows_ = K.nRows_;
			nCols_ = K.nCols_;
		}
		// assignment operator
		PinholeCameraIntrinsics& operator=(const PinholeCameraIntrinsics& K)
		{
			if (this != &K)
			{
				fx_ = K.fx_;
				fy_ = K.fy_;
				cx_ = K.cx_;
				cy_ = K.cy_;
				memcpy(distortion_parameters_, K.distortion_parameters_, sizeof(double) * 5);
				nRows_ = K.nRows_;
				nCols_ = K.nCols_;
			}
			return *this;
		}
		double fx() const { return fx_; }
		double fy() const { return fy_; }
		double cx() const { return cx_; }
		double cy() const { return cy_; }

		double& fx() { return fx_; }
		double& fy() { return fy_; }
		double& cx() { return cx_; }
		double& cy() { return cy_; }

		int nRows() const { return nRows_; }
		int nCols() const { return nCols_; }

		void SetDistortion(const double* InputDistortion)
		{
			memcpy(distortion_parameters_, InputDistortion, sizeof(double) * 5);
		}
		void SetImageSize(const int nRows, const int nCols)
		{
			nRows_ = nRows;
			nCols_ = nCols;
		}
		cv::Mat ToMatrix() const {
			cv::Mat K = cv::Mat::zeros(3, 3, CV_64FC1);
			K.at<double>(0, 0) = fx_;
			K.at<double>(1, 1) = fy_;
			K.at<double>(2, 2) = 1.0;
			K.at<double>(0, 2) = cx_;
			K.at<double>(1, 2) = cy_;
			return K.clone();
		}
		template <typename T>
		cv::Point3_<T> ProjectToSpace(const cv::Point_<T>& InputPt, T Depth) const
		{
			cv::Point3_<T> OutputPt;
			OutputPt.x = (InputPt.x - cx_) / fx_ * Depth;
			OutputPt.y = (InputPt.y - cy_) / fy_ * Depth;
			OutputPt.z = Depth;
			return OutputPt;
		}
		template <typename T>
		std::vector<cv::Point3_<T>> ProjectToSpace(const std::vector<cv::Point_<T>>& InputPt, std::vector<T> Depth) const
		{
			std::vector<cv::Point3_<T>> OutputPts;
			for (int i = 0; i < InputPt.size(); i++) {
				cv::Point3_<T> OutputPt;
				OutputPt.x = (InputPt[i].x - cx_) / fx_ * Depth[i];
				OutputPt.y = (InputPt[i].y - cy_) / fy_ * Depth[i];
				OutputPt.z = Depth[i];
				OutputPts.push_back(OutputPt);
			}
			return OutputPts;
		}
		template <typename T>
		cv::Point_<T> ProjectToImage(const cv::Point3_<T>& InputPt) const
		{
			cv::Point_<T> OutputPt;
			OutputPt.x = InputPt.x / InputPt.z * fx_ + cx_;
			OutputPt.y = InputPt.y / InputPt.z * fy_ + cy_;
			return OutputPt;
		}
		template <typename T>
		bool isOutsideImage(const cv::Point_<T>& InputPt) const
		{
			T x = InputPt.x;
			T y = InputPt.y;
			if (x <= 0 || x >= nCols_ - 1 || y <= 0 || y >= nRows_ - 1)
				return true;
			else
				return false;
		}
		void UndistortPoints(std::vector<cv::Point2f>& InputPts, std::vector<cv::Point2f>& OutputPts) const
		{
			if (InputPts.empty()) return;
			if (distortion_parameters_[0] * distortion_parameters_[0] +
				distortion_parameters_[1] * distortion_parameters_[1] +
				distortion_parameters_[2] * distortion_parameters_[2] +
				distortion_parameters_[3] * distortion_parameters_[3] +
				distortion_parameters_[4] * distortion_parameters_[4] > 1e-10)
			{
				cv::Mat InputMat(InputPts.size(), 1, CV_64FC2);
				for (size_t i = 0; i < InputPts.size(); i++)
				{
					InputMat.at<cv::Point2d>(i, 0).x = InputPts[i].x;
					InputMat.at<cv::Point2d>(i, 0).y = InputPts[i].y;
				}
				cv::Mat K = cv::Mat::zeros(3, 3, CV_64FC1);
				K.at<double>(0, 0) = fx_;
				K.at<double>(1, 1) = fy_;
				K.at<double>(2, 2) = 1.0;
				K.at<double>(0, 2) = cx_;
				K.at<double>(1, 2) = cy_;
				cv::Mat distCoeffs;
				if (std::abs(distortion_parameters_[4]) > 1e-10)
				{
					distCoeffs.create(1, 5, CV_64FC1);
					for (int i = 0; i < 5; i++)
						distCoeffs.at<double>(0,i) = distortion_parameters_[i];
				}
				else
				{
					distCoeffs.create(1, 4, CV_64FC1);
					for (int i = 0; i < 4; i++)
						distCoeffs.at<double>(0, i) = distortion_parameters_[i];
				}
				cv::Mat OutputMat(InputPts.size(), 1, CV_64FC2);

				cv::undistortPoints(InputMat, OutputMat, K, distCoeffs);
				OutputPts.resize(InputPts.size());
				for (size_t i = 0; i < OutputPts.size(); i++)
				{
					OutputPts[i].x = OutputMat.at<cv::Point2d>(i, 0).x * fx_ + cx_;
					OutputPts[i].y = OutputMat.at<cv::Point2d>(i, 0).y * fy_ + cy_;
				}
			}
			else
			{
				// do nothing
				OutputPts = InputPts;
			}
		}
	private:
		double fx_, fy_, cx_, cy_;
		// radial: [0,1,4], tangential: [2,3]
		double distortion_parameters_[5];
		// size of image
		int nRows_, nCols_;
	};
} // namespace CLOUDREAM

#endif
