#include<opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
# include <stdio.h>
# include <stdlib.h>
#include "opencv2/imgcodecs.hpp"
#include "opencv/highgui.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "math.h"
#include <vector>
#include "Marker.hpp"
#include "MarkerDetector.hpp"
#include "TinyLA.hpp"
#include "CameraCalibration.hpp"
#include "BGRAVideoFrame.h"
#include "GeometryTypes.hpp"

#include"cube.h"

using namespace std;
using namespace cv;

void generateIdMatrix(vector<cv::Mat>& normalidMatrix);
void cal_id_matrix(Mat& marker, Mat& id_matrix);
void PNP_RT(vector<Point3f>& m_cal3d, Marker& m, Mat& camMatrix, Mat& distCoeff);
void readParameters(Mat& camMatrix, Mat& distCoeff, float& length);// parameters.xml
void calcWorldPoint(float length, vector<cv::Point3f>& markerCorner);

const int idNum = 10;


int main()
{
    int badFrame = 0;

    Mat camMatrix, distCoeff;
    float length = 0;
    vector<cv::Mat> normalidMatrix;
    vector<cv::Point3f> markerCorner;
    readParameters(camMatrix, distCoeff, length);
    generateIdMatrix(normalidMatrix);
    calcWorldPoint(length, markerCorner);

    cv::VideoCapture capture(1); // 打开摄像头
    if (capture.isOpened() == false)
    {
        cout << "Failed to read the camera !" << endl;

    }
    cv::Mat imageTemp;
    capture >> imageTemp;
    /*int width = imageTemp.cols;
    int height = imageTemp.rows;*/
    namedWindow("Detect Result");
    imshow("Detect Result", imageTemp);
    waitKey(100);
    while (1)
    {
        capture >> imageTemp;
        //Mat imageTemp = imread("test.jpg");
        Mat dstImage, grayImage;
        cvtColor(imageTemp, grayImage, CV_BGR2GRAY);
        threshold(grayImage, dstImage, 150, 255, CV_THRESH_BINARY);
        vector<vector<cv::Point>> contours;
        vector<vector<cv::Point>> newContours;
        vector<Vec4i> hierarchy;

        cv::findContours(
                dstImage,
                contours,
                hierarchy,
                cv::RETR_CCOMP,
                cv::CHAIN_APPROX_NONE
        );

        Scalar color(0, 255, 0);


        int minContour = 100;
        for (int i = 0; i < contours.size(); i++)//去掉小边缘
        {
            int contourSize = contours[i].size();
            if (contourSize > minContour)
            {
                newContours.push_back(contours[i]);
            }
        }
        /*for (int i = 0; i < newContours.size(); i++)
        {
            vector<vector<cv::Point>> singleContours;
            singleContours.push_back(newContours[i]);
            drawContours(drawImage, singleContours, -1, color, 0, 8);
            imshow("边缘图像", drawImage);
            imwrite("边缘图像.jpg", drawImage);
            singleContours.clear();
        }*/
        /*drawContours(oriImage, newContours, -1, color, CV_FILLED, 8);
        imshow("边缘图像", oriImage);
        imwrite("边缘图像.jpg", oriImage);*/
        std::vector<Marker> possibleMarkers;// For each contour, analyze if it is a parallelepiped likely to be the marker
        Marker detectedMarkers;
        int m_minContourLengthAllowed = 60;
        std::vector<cv::Point>  approxCurve;



        for (size_t i = 0; i < newContours.size(); i++)
        {
            // Approximate to a polygon
            //double eps = newContours[i].size() * 0.05;
            cv::approxPolyDP(newContours[i], approxCurve, 5, true);

            // We interested only in polygons that contains only four points
            if (approxCurve.size() != 4)
                continue;

            // And they have to be convex
            if (!cv::isContourConvex(approxCurve))
                continue;

            // Ensure that the distance between consecutive points is large enough
            float minDist = std::numeric_limits<float>::max();

            for (int j = 0; j < 4; j++)
            {
                cv::Point side = approxCurve[j] - approxCurve[(j + 1) % 4];
                float squaredSideLength = side.dot(side);
                minDist = std::min(minDist, squaredSideLength);
            }

            // Check that distance is not very small
            if (minDist < m_minContourLengthAllowed)
                continue;

            /*vector<vector<cv::Point>> singleContours;
            singleContours.push_back(newContours[i]);
            drawContours(drawImage1, singleContours, -1, color, 0, 8);
            imshow("四边形长度一定的边缘", drawImage1);
            imwrite("四边形长度一定的边缘.jpg", drawImage1);
            singleContours.clear();*/

            // All tests are passed. Save marker candidate:
            Marker m;

            for (int j = 0; j < 4; j++)
                m.points.push_back(cv::Point2f(approxCurve[j].x, approxCurve[j].y));

            // Sort the points in anti-clockwise order
            // Trace a line between the first and second point.
            // If the third point is at the right side, then the points are anti-clockwise
            cv::Point v1 = m.points[1] - m.points[0];
            cv::Point v2 = m.points[2] - m.points[0];

            double o = (v1.x * v2.y) - (v1.y * v2.x);

            if (o < 0.0)		 //if the third point is in the left side, then sort in anti-clockwise order
                std::swap(m.points[1], m.points[3]);

            possibleMarkers.push_back(m);
        }

        //找最大值，选用周长最大的marker点当作detectedMarkers
        //周长做大的标记对应着实际中距离摄像头最近的marker标记
        if (possibleMarkers.size() == 0) // 如果没有，下一个
        {
            badFrame++;
            continue;
        }
        if (possibleMarkers.size() > 1) // 如果多个，选一个面积最大的
        {
            float maxP = std::numeric_limits<float>::min();
            int j=0;
            for (int i = 0; i < possibleMarkers.size(); i++)
            {
                float p = perimeter(possibleMarkers[i].points);
                if (p > maxP) {
                    maxP = p;
                    j = i;
                }
            }
            detectedMarkers = possibleMarkers[j];
        }
        else {  // 只有一个，那就是他了～
            detectedMarkers = possibleMarkers[0];
        }

        for (int i = 0; i < detectedMarkers.points.size(); i++)
        {
            cout << detectedMarkers.points[i] << endl;
        }


        cv::Size markerSize(100, 100);// 定义一个 100*100的marker 作为投射的目标
        std::vector<cv::Point2f> m_markerCorners2d;
        cv::Mat canonicalMarkerImage;

        m_markerCorners2d.push_back(cv::Point2f(markerSize.width - 1, 0));
        m_markerCorners2d.push_back(cv::Point2f(markerSize.width - 1, markerSize.height - 1));
        m_markerCorners2d.push_back(cv::Point2f(0, markerSize.height - 1));
        m_markerCorners2d.push_back(cv::Point2f(0,0));
        // 透视变换，成像投影到一个新的视平面。计算出透视变换矩阵M(3*3)
        cv::Mat markerTransform = cv::getPerspectiveTransform(detectedMarkers.points, m_markerCorners2d);
        // 投射变换函数, 将拍到的图片结合R，T矩阵
//        void cv::warpPerspective(
//                cv::InputArray src, // 输入图像
//                cv::OutputArray dst, // 输出图像
//                cv::InputArray M, // 3x3 变换矩阵
//                cv::Size dsize, // 目标图像大小
//                int flags = cv::INTER_LINEAR, // 插值方法
//                int borderMode = cv::BORDER_CONSTANT, // 外推方法
//                const cv::Scalar& borderValue = cv::Scalar() //常量边界时使用);
        cv::warpPerspective(grayImage, canonicalMarkerImage, markerTransform, markerSize);

        threshold(canonicalMarkerImage, canonicalMarkerImage, 160, 255, THRESH_BINARY | THRESH_OTSU); // 二值化

        // 投射变换结束，接下来识别marker标记中的汉明码
        Mat idMatrix = Mat::zeros(7, 7, CV_32S);
        Mat diff_idMatrix[4];
        for (int j = 0; j < 4; j++)
        {
            // 矩阵转置

            transpose(canonicalMarkerImage, canonicalMarkerImage);
            //0: 沿X轴翻转； >0: 沿Y轴翻转； <0: 沿X轴和Y轴翻转
            //flip(canonicalMarkerImage, canonicalMarkerImage, 0);
            cal_id_matrix(canonicalMarkerImage, idMatrix);
            diff_idMatrix[j] = idMatrix.clone();
            flip(canonicalMarkerImage, canonicalMarkerImage, 1); // 图像的反转采用flip函数实现,该函数能够实现图像在水平方向,垂直方向和水平垂直方向的旋转,
        }
        // 识别汉明码
        //=============?? 4是什么意思
        bool isMatch = false;
        for (int i = 0; i < idNum; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Mat countMatrix;

                countMatrix = diff_idMatrix[j] - normalidMatrix[i];

                int count = countNonZero(countMatrix);
                if (count == 0)
                {
                    isMatch = true;
                    badFrame = 0;
                    detectedMarkers.id = i;
                    break;
                }
            }
            if (isMatch == true)
                break;
        }
        if (isMatch == false)
        {
            badFrame++;
            continue;
        }
        else {
            cv::TermCriteria termCriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.01);
            cv::cornerSubPix(grayImage, detectedMarkers.points, cvSize(5, 5), cvSize(-1, -1), termCriteria);
        }
        if (badFrame >= 10)
        {
            cout << "missing target!" << endl;
        }
        // 调用PnP函数，计算出相机坐标系相对于标记坐标系的旋转，平移矩阵
        PNP_RT(markerCorner, detectedMarkers, camMatrix, distCoeff);
        // 画图部分，显示出来相应的轮廓和点的坐标
        Mat frame = imageTemp.clone();
        for (int i = 0; i < detectedMarkers.points.size(); i++)
        {
            cv::circle(frame, detectedMarkers.points[i], 3, cv::Scalar(255, 0, 0, 255), 5);
            if (i != 3)
                cv::line(frame, detectedMarkers.points[i], detectedMarkers.points[i + 1], cv::Scalar(0, 255, 0, 255),3);
            else
                cv::line(frame, detectedMarkers.points[i], detectedMarkers.points[0], cv::Scalar(0, 255, 0, 255),3);
        }
        std::string id = to_string(detectedMarkers.id);
        std::string text0 = "id=" + id;
        Point2f org;
        org.x = (detectedMarkers.points[1].x + detectedMarkers.points[2].x) / 2;
        org.y = (detectedMarkers.points[0].y + detectedMarkers.points[1].y) / 2;
        int font_face = cv::FONT_HERSHEY_COMPLEX;
        double font_scale = 0.3;
        int thickness = 0.3;
        putText(frame, text0, org, font_face, font_scale, cv::Scalar(0, 0, 255, 255));
        //获取文本框的长宽
        std::string text1 = "x=" + to_string(detectedMarkers.transformation.t().data[0]);
        std::string text2 = "y=" + to_string(detectedMarkers.transformation.t().data[1]);
        std::string text3 = "z=" + to_string(detectedMarkers.transformation.t().data[2]);
        org.y = org.y + 10;
        putText(frame, text1, org, font_face, font_scale, cv::Scalar(0, 0, 255, 255));
        org.y = org.y + 10;
        putText(frame, text2, org, font_face, font_scale, cv::Scalar(0, 0, 255, 255));
        org.y = org.y + 10;
        putText(frame, text3, org, font_face, font_scale, cv::Scalar(0, 0, 255, 255));
        // 将每一帧计算出来的坐标保存到TXT文件中，以便于后续的处理评估
        ofstream outfile;
        outfile.open("/Users/zhaocongyang/Documents/shiyan_marker_PNP/canshu.txt", ios::binary | ios::app | ios::in | ios::out);
        outfile<<"X = "<<detectedMarkers.transformation.t().data[0]<<"   ";
        outfile<<"Y = "<<detectedMarkers.transformation.t().data[1]<<"   ";
        outfile<<"Z = "<<detectedMarkers.transformation.t().data[2]<<"\n";
        outfile.close();//关闭文件，保存文件

        //namedWindow("Detect Result");
        imshow("Detect Result", frame);
        waitKey(100);
        //imwrite("Final Result.jpg", frame);

    }
}

void cal_id_matrix(Mat& marker, Mat& id_matrix)
{
    int cellSize = marker.rows / 7;

    for (int y = 0; y < 7; y++)
    {
        //int inc = 10;
        for (int x = 0; x < 7; x++)
        {
            int cellX = x*cellSize;
            int cellY = y*cellSize;
            Mat cell = marker(Rect(cellX, cellY, cellSize, cellSize));

            int nZ = countNonZero(cell);//统计区域内非0的个数。

            if (nZ >(cellSize*cellSize) / 2)
            {
                id_matrix.at<int>(y, x) = 1;
            }
            else {
                id_matrix.at<int>(y, x) = 0;
            }
        }

    }
}

// PnP 函数，实现单目marker标记的旋转矩阵和平移矩阵的解算
//输入：marker标记的实际坐标 m_cal3d， 和marker的标记 m
//输出；无返回值，执行结束后marker类的 transformation 成员会被赋值
void PNP_RT(vector<Point3f>& m_cal3d, Marker& m, Mat& camMatrix, Mat& distCoeff)
{
    Mat raux, taux;// PnP 的输出，旋转矩阵和平移矩阵
    Mat Rvec; // 用于存储 mat 转换类型之后的数据
    Mat_<float> Tvec;//Mat_<float>对应的是CV_32F
    solvePnP(m_cal3d, m.points, camMatrix, distCoeff, raux, taux);
    raux.convertTo(Rvec, CV_32F);//转换Mat的保存类型，输出Rvec
    taux.convertTo(Tvec, CV_32F);
    Mat_<float> rotMat(3, 3);
    Rodrigues(Rvec, rotMat);//罗德里格斯变换对旋转向量和旋转矩阵进行转换，输出旋转矩阵rotMat
    cout << " m.transformation.r()" << endl;
    // 通过 rotMat 给m的成员赋值
    for (int col = 0; col < 3; col++)
    {

        for (int row = 0; row < 3; row++)
        {
            m.transformation.r().mat[row][col] = rotMat(row, col);//copy rotation component
            cout << m.transformation.r().mat[row][col] << " ";
        }

        m.transformation.t().data[col] = Tvec(col);//copy translation component//复制位移向量到标识类的变量
        cout << m.transformation.t().data[col] << endl;
        cout << endl;
    }
    cout << " m.transformation.t()" << endl;
}

void readParameters(Mat& camMatrix, Mat& distCoeff, float& length)
{
    FileStorage parameters("/Users/zhaocongyang/Documents/shiyan_marker_PNP/parameters.xml", FileStorage::READ);
    if (!parameters.isOpened())
    {
        cerr << "failed to open parameters.xml" << endl;
    }
    parameters["camMatrix"] >> camMatrix;
    parameters["distCoeff"] >> distCoeff;
    parameters["length"] >> length;

    parameters.release();
}

void calcWorldPoint(float length, vector<cv::Point3f>& markerCorner)
{
    //topright,bottomright,bottomleft,topleft
    Point3f temp;
    temp.x = length;
    temp.y = 0;
    temp.z = 0;
    markerCorner.push_back(temp);

    temp.x = length;
    temp.y = length;
    markerCorner.push_back(temp);

    temp.x = 0;
    temp.y = length;
    markerCorner.push_back(temp);

    temp.x = 0;
    temp.y = 0;
    markerCorner.push_back(temp);
}

void generateIdMatrix(vector<cv::Mat>& normalidMatrix)
{
    Mat M;
    M = (Mat_<int>(7, 7) << 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 0, 0, 0,
            0, 1, 1, 1, 1, 0, 0,
            0, 1, 1, 1, 1, 0, 0,
            0, 0, 0, 0, 1, 0, 0,
            0, 1, 1, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0);
    normalidMatrix.push_back(M);//1
    M = (Mat_<int>(7, 7) << 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 0,
            0, 1, 0, 1, 1, 1, 0,
            0, 1, 0, 1, 1, 0, 0,
            0, 1, 0, 1, 1, 0, 0,
            0, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0);
    normalidMatrix.push_back(M);//2
    M = (Mat_<int>(7, 7) << 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 1, 0, 0,
            0, 1, 0, 1, 0, 0, 0,
            0, 1, 0, 1, 1, 0, 0,
            0, 1, 0, 1, 1, 0, 0,
            0, 1, 1, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0);
    normalidMatrix.push_back(M);//3
    M = (Mat_<int>(7, 7) << 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 1, 1, 0, 0,
            0, 0, 1, 1, 1, 0, 0,
            0, 0, 1, 1, 1, 0, 0,
            0, 0, 0, 0, 1, 0, 0,
            0, 1, 1, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0);
    normalidMatrix.push_back(M);//4
    M = (Mat_<int>(7, 7) << 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 1, 1, 0, 0,
            0, 0, 0, 1, 1, 0, 0,
            0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 0, 1, 1, 0,
            0, 1, 1, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0);
    normalidMatrix.push_back(M);//5
    M = (Mat_<int>(7, 7) << 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 1, 1, 0,
            0, 0, 1, 0, 1, 1, 0,
            0, 1, 0, 1, 0, 1, 0,
            0, 1, 0, 1, 1, 0, 0,
            0, 1, 1, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0);
    normalidMatrix.push_back(M);//6
    M = (Mat_<int>(7, 7) << 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 1, 1, 0, 0,
            0, 1, 0, 1, 1, 0, 0,
            0, 1, 1, 0, 0, 0, 0,
            0, 1, 1, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0);
    normalidMatrix.push_back(M);//7
    M = (Mat_<int>(7, 7) << 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 0, 1, 0,
            0, 1, 1, 0, 1, 0, 0,
            0, 1, 1, 0, 1, 0, 0,
            0, 1, 0, 1, 1, 0, 0,
            0, 1, 1, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0);
    normalidMatrix.push_back(M);//8
    M = (Mat_<int>(7, 7) << 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 1, 1, 0, 0,
            0, 1, 1, 1, 0, 1, 0,
            0, 1, 0, 1, 0, 1, 0,
            0, 1, 1, 1, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0);
    normalidMatrix.push_back(M);//9
    M = (Mat_<int>(7, 7) << 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 1, 1, 1, 0,
            0, 1, 0, 1, 0, 0, 0,
            0, 1, 1, 0, 1, 0, 0,
            0, 1, 1, 1, 1, 0, 0,
            0, 1, 1, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0);
    normalidMatrix.push_back(M);//10
}