#include "kcalchist.h"


#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using std::cin;
using std::cout;
using std::endl;

KCalcHist::KCalcHist(cv::Mat & img)
    :m_mImage(img)
{
}

KCalcHist::KCalcHist(const char *filename)
//    : m_sInFileName(filename)
{
    if(nullptr == filename){ cout << "CalcGrayHistgram:Input filename is empty!" << endl; }
    m_mImage = cv::imread(filename);
}

bool KCalcHist::CalcGrayHistgram(cv::MatND &hist)
{
    if(!m_mImage.data){
         cout << "CalcGrayHistgram:InputFile read failed!"; return false;
    }

    cv::Mat grayImg;
    if(m_mImage.channels() == 3){
        cv::cvtColor(m_mImage,grayImg,CV_BGR2GRAY);
    }else{
        if(m_mImage.channels() != 1){
            cout << "CalcGrayHistgram:InputFile has invalid channels!"; return false;
        }else grayImg = m_mImage;
    }
    int histSize[1] = {256};
    int chanNum[1] = {0};
    float rangesItem[2] = {0.f, 255.f};
    const float *ranges[1] = {rangesItem};
    cv::calcHist(&grayImg, 1, chanNum, cv::Mat(), hist, 1, histSize, ranges);
    return true;
}

void KCalcHist::ShowGrayHistgram(cv::MatND &hist, const char * wndName)
{
    cv::String sWndName("");
    if(nullptr == wndName) sWndName += "Histgram";
    else sWndName += wndName;

    cv::Mat histImg(256, 256, CV_8UC1, cv::Scalar(255));

    double minValue(0.);
    double maxValue(0.);

    cv::minMaxLoc(hist,&minValue,&maxValue,NULL,NULL);

    double span = maxValue - minValue;

    for(int index = 0;index < 256;++index){

        float pix = hist.at<float>(index);
        unsigned char pixIntensity = cv::saturate_cast<unsigned char>((pix - minValue) * 256 / span);
        //cout<<pix;
        cv::line(histImg,cv::Point(index,256),cv::Point(index,256-pixIntensity),cv::Scalar_<unsigned char>::all(0));
    }

    cv::namedWindow(sWndName);
    cv::imshow(sWndName, histImg);
}

void KCalcHist::NormalizeHistgram(cv::MatND &hist, cv::Mat& norHist)
{
    double minValue(0.);
    double maxValue(0.);

    cv::minMaxLoc(hist,&minValue,&maxValue,NULL,NULL);

    double span = maxValue - minValue;

    norHist.create(1, 256, CV_8U);

    for(int index = 0;index < 256;++index){

        float pix = hist.at<float>(index);
        norHist.at<unsigned char>(index) = cv::saturate_cast<unsigned char>((pix - minValue) * 256 / span);
    }
}
