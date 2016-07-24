#ifndef KCALCHIST_H
#define KCALCHIST_H

#include <opencv2/core/core.hpp>

class KCalcHist
{
public:
    KCalcHist(cv::Mat&);
    KCalcHist(const char * = nullptr);
    bool CalcGrayHistgram(cv::MatND&);
    void ShowGrayHistgram(cv::MatND&, const char * = nullptr);
    void NormalizeHistgram(cv::MatND&, cv::Mat&);
private:
    cv::Mat m_cmInImage;
    //cv::String m_sInFileName;
    cv::Mat m_mImage;
};

#endif // KCALCHIST_H
