#include<string>
#include<time.h>
#include<windows.h>
#include<iostream>
#include<fstream>
#include<opencv2/opencv.hpp>
#include "depth_denoising.h"
#include "cloudream_pinhole_camera.h"

using namespace std;
using namespace cv;

void WriteImgList(const string fileName, const string pathPrefix, const int start, const int end) {
	ofstream ofs;
	ofs.open(fileName);
	if (ofs.is_open()) {
		for (int i = start; i <= end; i++) {
			stringstream ss;
			ss << i << endl;
			string out;
			ss >> out;
			//while (out.length() < 4) {
			//	out = "0" + out;
			//}
			ofs << pathPrefix + out + ".png" << endl;
		}
		ofs.close();
	}
	else {
		cout << "writeImgList: open file error!" << endl;
		exit(0);
	}
}

void GetImgsPath(string listDir, vector<string>& imgsPath) {
	//read it to the storage	
	ifstream readPic(listDir);
	if (!readPic)
	{
		cout << "error: Cannot open the dir!" << endl;
		exit(0);
	}
	string tmpStr;
	while (getline(readPic, tmpStr))
	{
		imgsPath.push_back(tmpStr);
	}
	readPic.close();
}


void SaveCloud(std::vector<Point3f> pts, string fileName) {
	int count = 0;
	for (int i = 0; i < pts.size(); i++)
	{
		cv::Point3f p = pts[i];
		if (p.x != -std::numeric_limits<float>::infinity() && p.y != -std::numeric_limits<float>::infinity() && p.z != -std::numeric_limits<float>::infinity() && abs(p.z) > 1e-6)
			{
				count++;
			}
	}

	ofstream ofs;
	ofs.open(fileName);
	if (ofs.is_open()) {
		ofs << "ply\n";
		ofs << "format ascii 1.0\n";
		ofs << "element vertex " << count << "\n";
		ofs << "property float x\n";
		ofs << "property float y\n";
		ofs << "property float z\n";
		ofs << "property float nx\n";
		ofs << "property float ny\n";
		ofs << "property float nz\n";
		ofs << "property uchar diffuse_red\n";
		ofs << "property uchar diffuse_green\n";
		ofs << "property uchar diffuse_blue\n";
		ofs << "property uchar alpha\n";
		ofs << "end_header\n";
		for (int i = 0; i < pts.size(); i++)
		{
			cv::Point3f p = pts[i];
			if (p.x != -std::numeric_limits<float>::infinity() && p.y != -std::numeric_limits<float>::infinity() && p.z != -std::numeric_limits<float>::infinity() && abs(p.z) > 1e-6)
			{
				float cameraX = static_cast<float>(p.x);
				float cameraY = static_cast<float>(p.y);
				float cameraZ = static_cast<float>(p.z);
				ofs << cameraX << " " << cameraY << " " << cameraZ << " ";
				ofs << "0 0 0 ";
				ofs << "255 255 255 ";
				ofs << "255" << endl;
			}
		}
		ofs.close();
		cout << "depthtocloud ok!" << endl;

	}

}

int  main() {
	const int WIDTH = 640;
	const int HEIGHT = 480;

	for (int mn = 2; mn <= 30; mn++) {
		WriteImgList(".\\inDepth.txt", "H:/data/depthf200/depth_", 2, mn);
		vector<string> depthImgsPath;
		GetImgsPath(".\\inDepth.txt", depthImgsPath);

		DepthDenoising dd(WIDTH, HEIGHT);
		std::vector<std::vector<float>> res;

		//float k[4] = { 364.8929, 364.9805, 254.4063, 210.3299 };
		//double distortion[5] = { 0.0915, -0.3075, 0, 0, 0.1541 };
		float k[4] = { 463.889, 463.889, 320, 240 };
		double distortion[5] = { 0 };
		CLOUDREAM::PinholeCameraIntrinsics pci;
		pci.fx() = k[0];
		pci.fy() = k[1];
		pci.cx() = k[2];
		pci.cy() = k[3];
		pci.SetDistortion(distortion);
		pci.SetImageSize(HEIGHT, WIDTH);

		clock_t start, end;
		start = clock();
		double sum = 0;
		for (int i = 0; i < depthImgsPath.size(); i++) {
			clock_t t1, t2;
			t1 = clock();
			Mat tdepth = imread(depthImgsPath[i], CV_LOAD_IMAGE_UNCHANGED);
			t2 = clock();
			sum += static_cast<double>(t2 - t1);
			CV_Assert(!tdepth.empty() && (tdepth.type() == CV_16UC1));
			if (i < depthImgsPath.size() - 1)
				dd.setCurDepth(tdepth, false);
			else if (i == depthImgsPath.size() - 1) {
				dd.setCurDepth(tdepth, true);
			}
			//dd.dealCurDepth_GPU();	
			dd.dealCurDepth_CPU();

			//std::vector<float> de = dd.downloadCurDepth();
			//std::vector<cv::Point2f> pos, pos_u;
			//for (int r = 0; r < HEIGHT; r++) {
			//	for (int c = 0; c < WIDTH; c++) {
			//		pos.push_back(cv::Point2f(c, r));
			//	}
			//}
			//pci.UndistortPoints(pos, pos_u);
			//std::vector<cv::Point3f> cloud_cur;
			//cloud_cur = pci.ProjectToSpace(pos_u, de);
			//stringstream ss;
			//ss << i << endl;
			//string out;
			//ss >> out;
			//SaveCloud(cloud_cur, "cloud_" + string(out) + ".ply");

			//std::vector<cv::Point3f> cloud_cur;
			//float max = -1.0f, min = 1000.9f;
			//for (int r = 0; r < HEIGHT; r++) {
			//	for (int c = 0; c < WIDTH; c++) {
			//		unsigned index = r * WIDTH + c;
			//		if (tdepth.at<unsigned short>(r, c) / 1000.0f > max)
			//			max = tdepth.at<unsigned short>(r, c) / 1000.0f;
			//		if (tdepth.at<unsigned short>(r, c) / 1000.0f < min)
			//			min = tdepth.at<unsigned short>(r, c) / 1000.0f;
			//		cv::Point3f pt = pci.ProjectToSpace(cv::Point2f(c, r), tdepth.at<unsigned short>(r, c) / 1000.0f);
			//		cloud_cur.push_back(pt);
			//	}
			//	
			//}
			//cout << "max:" << max << " ,   min:" << min << endl;
			//stringstream ss;
			//ss << i << endl;
			//string out;
			//ss >> out;
			//SaveCloud(cloud_cur, "cloud_" + string(out) + ".ply");
		}
		//dd.dealAllDepth_GPU();
		res = dd.downloadDenoisedRes();

		end = clock();
		double dur = (double)(end - start);
		cout << "¶ÁÍ¼ºÄÊ±£º" << sum << " ºÁÃë" << endl;
		cout << "½µÔëºÄÊ±:" << dur - sum << " ºÁÃë" << endl;

		std::vector<cv::Point2f> without_noise_pts, with_noise_pts;
		std::vector<cv::Point2f> without_noise_pts_u, with_noise_pts_u;
		std::vector<float> depth_without_noise_pts, depth_with_noise_pts;
		std::vector<cv::Point3f> cloud_without_noise, cloud_with_noise;
		for (int r = 0; r < HEIGHT; r++) {
			for (int c = 0; c < WIDTH; c++) {
				unsigned index = r * WIDTH + c;
				if (res[index].size() == 1) {
					without_noise_pts.push_back(Point2f(c, r));
					depth_without_noise_pts.push_back(res[index][0]);

					with_noise_pts.push_back(Point2f(c, r));
					depth_with_noise_pts.push_back(res[index][0]);
				}
				else if (res[index].size() > 1) {
					with_noise_pts.push_back(Point2f(c, r));
					depth_with_noise_pts.push_back(res[index][res[index].size() / 2]);
				}
				else if (res[index].size() < 1) {
					cout << "row:" << r << "  ,col:" << c << ", cannot find depth!" << endl;
					return 0;
				}
			}
		}


		pci.UndistortPoints(without_noise_pts, without_noise_pts_u);
		pci.UndistortPoints(with_noise_pts, with_noise_pts_u);

		for (int i = 0; i < without_noise_pts_u.size(); i++) {
			cv::Point3f pt = pci.ProjectToSpace(without_noise_pts_u[i], depth_without_noise_pts[i]);
			cloud_without_noise.push_back(pt);
		}
		for (int i = 0; i < with_noise_pts_u.size(); i++) {
			cv::Point3f pt = pci.ProjectToSpace(with_noise_pts_u[i], depth_with_noise_pts[i]);
			cloud_with_noise.push_back(pt);
		}
		stringstream ss;
		ss << mn << endl;
		string out;
		ss >> out;
		SaveCloud(cloud_without_noise, "withoutnoise" + out +".ply");
		SaveCloud(cloud_with_noise, "withnoise" + out + ".ply");
	}	
	return 0;
}





