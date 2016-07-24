#ifndef KUTILITY_H
#define KUTILITY_H

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/ml.hpp>

class KUtility
{
public:
    static double ThresOTSU(cv::Mat);
    static int SplitNumber(cv::Mat, std::vector<int>&, std::vector<int>&);
    static bool RectangleNumber(cv::Mat, std::vector<int>&, std::vector<int>&);
    static void GetImgListByDir(std::string, std::vector<std::string>&);
    static void ExtractFeatures(std::string, std::string, std::vector<int>&, std::vector<std::vector<float>>&);
    static void ExtractFeaturesLIBSVM(std::string, std::string);
    static std::vector<float> GetFeatrueVec(cv::Mat);
    static void TrainProcess(std::string, std::string, cv::Ptr<cv::ml::SVM>&);
    static void PredictProcess(std::string, std::string, std::string);
    static void BuildModelAuto(cv::Ptr<cv::ml::SVM>&, std::string, cv::Mat, cv::Mat);
    static void ExtractFeatures(std::string, std::string, std::vector<std::vector<float>>&);
    static void RemoveSmall(cv::Mat, int = 2);
    static void CreatFileUseLIBSVM(std::string, std::string);
    static void GetLBPVec(cv::Mat,std::vector<float>&);
    static void FeatureScale(std::string, std::string);
    static int GetFeatureDims(std::string);
    static void CreateScaleParam(std::string, std::string, float = -1.f, float = 1.f);
private:
    KUtility();
    KUtility(const KUtility&);
    KUtility& operator = (const KUtility&);
};

#endif // KUTILITY_H
