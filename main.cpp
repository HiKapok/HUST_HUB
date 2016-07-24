#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "kcalchist.h"
#include "kutility.h"

using namespace std;

int main(int argc, char *argv[])
{
    //KUtility::CreatFileUseLIBSVM("D:\\code","D:\\pre");
    KUtility::CreateScaleParam("D:\\code\\train_data.txt","D:\\code\\param.txt");
    KUtility::FeatureScale("D:\\code\\predict_data.txt","D:\\code\\param.txt");
    //KUtility::PredictProcess("D:\\code","D:\\code\\model.txt","D:\\pre");
//    cv::Mat temp;// = cv::imread(cv::String("D:\\code\\randomImage3.jpg").c_str());

//    std::vector<int> vecLabels;
//    std::vector<std::vector<float>> vecFeatures;
//    cout << "Hello World!" << endl;
//    KUtility::ExtractFeatures(std::string("D:\\code\\00413.jpg"),std::string("D:\\code\\00413.txt"), vecLabels, vecFeatures);
//    vector<string> imgLists;
//    KUtility::GetImgListByDir("D://code",imgLists);

////    cv::namedWindow("gray");

////    cv::cvtColor(temp,temp,CV_BGR2GRAY);

////    cv::imshow("gray",temp);

////    KCalcHist khist(temp);
////    cv::MatND histgramMat;
////    khist.CalcGrayHistgram(histgramMat);
////    //khist.ShowGrayHistgram(histgramMat);

////    //cv::Mat nHist;
////    //khist.NormalizeHistgram(histgramMat,nHist);
////    double thres=KUtility::ThresOTSU(temp);
////    //cout<<t<<endl;
////    //cv::threshold(nHist,temp,0,255,cv::THRESH_OTSU | cv::THRESH_BINARY);
////    //double thre = cv::threshold(temp,nHist,0,255,cv::THRESH_OTSU);
////    cv::threshold(temp,temp,thres,255,cv::THRESH_BINARY);
////    //cout<<thre<<endl;

////    vector<int> tempCol;
////    vector<int> tempRow;
////    if(-1 != KUtility::SplitNumber(temp,tempRow,tempCol))
////        KUtility::RectangleNumber(temp,tempRow,tempCol);

////    cv::imshow("gray",temp);

    cv::waitKey(0);

    return 0;
}
