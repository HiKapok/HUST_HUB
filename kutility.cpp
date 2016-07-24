#include "kutility.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <QString>
#include <QDebug>
#include <QDir>

#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <limits>

using std::string;
using std::vector;

KUtility::KUtility()
{
}

int KUtility::SplitNumber(cv::Mat _src, std::vector<int>& vecRowSplit, std::vector<int>& vecColSplit)
{
    int step = static_cast<int>(_src.step);
    cv::Size size = _src.size();

    int lastBlcakPoint = 0;

    for(int wIndex = 1; wIndex < size.width; wIndex++ )
    {
        const uchar* src = _src.ptr(0, wIndex);

        int blcakPoint = 0;
        for(int hIndex = 0; hIndex < size.height; hIndex++ ){
            if(src[step * hIndex]<128) ++blcakPoint;
        }

        if(blcakPoint > 0 && 0 == lastBlcakPoint) vecRowSplit.push_back(wIndex - 1);
        //if(0 == blcakPoint && lastBlcakPoint > 0) vecRowSplit.push_back(wIndex);

        if(0 == blcakPoint && lastBlcakPoint > 0){

            if(vecRowSplit[vecRowSplit.size()-1] + 3 > wIndex){
                //vecRowSplit.erase(vecRowSplit.rbegin().base());
                vecRowSplit.resize(vecRowSplit.size()-1);
                lastBlcakPoint = 0;
                continue;
            }

            vecRowSplit.push_back(wIndex);
            //lastBlcakPoint = 0;
            //break;
        }

        lastBlcakPoint = blcakPoint;
    }
    for(int hIndex = 1; hIndex < size.height; hIndex++ )
    {
        const uchar* src = _src.ptr() + step * hIndex;

        int blcakPoint = 0;
        for(int wIndex = 0; wIndex < size.width; wIndex++ ){
            if(src[wIndex]<128) ++blcakPoint;
        }

        if(blcakPoint > 0 && 0 == lastBlcakPoint) vecColSplit.push_back(hIndex - 1);
        if(0 == blcakPoint && lastBlcakPoint > 0){

            if(vecColSplit[vecColSplit.size()-1] + 10 > hIndex){
                //vecColSplit.erase(vecColSplit.rbegin().base());
                vecColSplit.resize(vecColSplit.size()-1);
                lastBlcakPoint = 0;
                continue;
            }

            vecColSplit.push_back(hIndex);
            //lastBlcakPoint = 0;
        }

        lastBlcakPoint = blcakPoint;
    }

    if( 2 != vecColSplit.size()) return -1;

    return vecRowSplit.size();
}

bool KUtility::RectangleNumber(cv::Mat _src, std::vector<int>& vecRowSplit, std::vector<int>& vecColSplit)
{
    if( 2 != vecColSplit.size()) return false;

    int up = *(vecColSplit.begin());
    int below = *(vecColSplit.rbegin());

    for(auto it = vecRowSplit.begin(); it != vecRowSplit.end(); ++it){
        cv::Point leftUp(*it++, up);
        cv::Point rightBelow(*it, below);

        cv::rectangle(_src,leftUp,rightBelow,cv::Scalar_<uchar>::all(130));
    }
    return true;
}

double KUtility::ThresOTSU(cv::Mat _src){
    const int N = 256;
    int i = 0, j = 0;
    cv::Size size = _src.size();
    int step = (int) _src.step;
    //vector<double> vecThres;

    int h[N] = {0};
    for( i = 0; i < size.height; i++ )
    {
        const uchar* src = _src.ptr() + step*i;
        j = 0;
        for( ; j < size.width; j++ )
            h[src[j]]++;
    }

    double mu = 0, scale = 1./(size.width*size.height);
    for( i = 0; i < N; i++ )
        mu += i*(double)h[i];

    mu *= scale;
    //cout<<mu<<endl;

    // the value 130 is estimated position
    // mu can be considered as the centroid
    int upBound = (static_cast<int>(mu) + 130) / 2;
    double mu1 = 0, q1 = 0;
    double max_sigma = 0, max_val = 0;

    int sigmaInt = 0;

    for( i = 0; i < N; i++ )
    {
        double p_i, q2, mu2, sigma;

        //foreground probability
        p_i = h[i]*scale;
        mu1 *= q1;
        q1 += p_i;
        q2 = 1. - q1;

        if( std::min(q1,q2) < FLT_EPSILON || std::max(q1,q2) > 1. - FLT_EPSILON )
            continue;

        mu1 = (mu1 + i*p_i)/q1;
        mu2 = (mu - q1*mu1)/q2;
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2);
        sigmaInt = static_cast<int>(sigma);
        //vecThres.push_back(sigma);
        //cout<<i<<":"<<sigmaInt<<" ";
        if( sigmaInt > max_sigma && i < upBound)
        {
            max_sigma = sigmaInt;
            max_val = i;
        }
    }

    return max_val;
}

void KUtility::GetImgListByDir(string dirName,vector<string> &imgList){
    QDir dir(dirName.c_str());
    if(!dir.exists()){
        std::cout << "wrong directory!" << std::endl;
        return;
    }
    QStringList filters;
    filters << "*.jpg";
    dir.setNameFilters(filters);
    QStringList fileLists = dir.entryList(QDir::Files);

   for (QStringList::const_iterator constIterator = fileLists.constBegin();
        constIterator != fileLists.constEnd();
          ++constIterator){
       QString tempStr = dir.path() + QString(QDir::separator()) + *constIterator;
       imgList.push_back(tempStr.toLocal8Bit().constData());

   }
}

void KUtility::ExtractFeatures(std::string file, std::string vecFileName, std::vector<int>& labelVec, std::vector<std::vector<float>>& featureVec)
{
    cv::Mat srcMat = cv::imread(file.c_str());

    cv::cvtColor(srcMat,srcMat,CV_BGR2GRAY);

    double thres=KUtility::ThresOTSU(srcMat);

    cv::threshold(srcMat,srcMat,thres,255,cv::THRESH_BINARY);
    vector<int> tempCol;
    vector<int> tempRow;

    char fname[_MAX_FNAME];
    _splitpath(file.c_str(), NULL, NULL, fname, NULL);
    string fileNameExt(fname);

    RemoveSmall(srcMat);
    //cv::imwrite(string("D:\\temp\\") + fname + string("-temp.jpg"),srcMat);

    if(-1 != KUtility::SplitNumber(srcMat,tempRow,tempCol))
    {
        if( 2 != tempCol.size()) return;

        if(fileNameExt.length() != tempRow.size() / 2) return;
        std::vector<char> tempLabel;
        for(auto tempChar : fileNameExt) tempLabel.push_back(tempChar);
        //std::vector<const char> tempLabel(fileNameExt.c_str(),fileNameExt.c_str()+fileNameExt.length());
        for(auto tempChar : tempLabel) labelVec.push_back(static_cast<int>(tempChar-static_cast<int>('0')));

        std::fstream fs(vecFileName.c_str(),std::ios_base::out | std::ios_base::app);

        int up = *(tempCol.begin()) + 1;
        int below = *(tempCol.rbegin());
        //int col_count = 0;

        //cv::namedWindow("gray");
        for(auto it = tempRow.begin(); it != tempRow.end(); ++it){
            int startCol = *(it++) + 1;
            int endCol = *it;
            cv::Mat numROI = srcMat(cv::Range(up,below),cv::Range(startCol,endCol));
            //cv::imshow("gray",numROI);
            std::vector<float> tempVec = GetFeatrueVec(numROI);
            featureVec.push_back(tempVec);
            //fs<<fileNameExt.at(col_count++);
            //fs<<" ";
            for(auto itemFeature : tempVec){
                std::ostringstream float2str("");
                float2str<<itemFeature;
                fs<<float2str.str()<<" ";
            }

            fs<<"\r\n";
        }

        fs.close();
    }
    //std::cout<<fileNameExt<<std::endl;
}

std::vector<float> KUtility::GetFeatrueVec(cv::Mat imag)
{
    int width = imag.cols;
    int height = imag.rows;
    //int step = imag.step;

    std::vector<float> vecFeatures;

    const int xBlocks = 2;
    const int yBlocks = 3;

    int xStep = width / xBlocks;
    int yStep = height / yBlocks;

    float *sumArray = new(std::nothrow) float[xBlocks * yBlocks];
    if(nullptr == sumArray) return vecFeatures;

    memset(sumArray,0,xBlocks * yBlocks * sizeof(float));

    const uchar* src = imag.ptr();
    int blackArea = 0;
    for(int iY = 0;iY < height;++iY){
        for(int iX = 0;iX < width;++iX){
            int xpos = iX / xStep;
            xpos = xpos > xBlocks - 1 ? xBlocks - 1 : xpos;
            int ypos = iY / yStep;
            ypos = ypos > yBlocks - 1 ? yBlocks - 1 : ypos;

            if(src[iY * width + iX] < 128) blackArea++;
            sumArray[ypos * xBlocks + xpos] += src[iY * width + iX];
        }
    }

    vecFeatures.push_back(width);
    vecFeatures.push_back(height);
    vecFeatures.push_back(width*1.f/height);
    vecFeatures.push_back(blackArea);

    for(int index = 0;index < xBlocks * yBlocks;++index){
        //sumArray[index] /= (xStep * yStep);
        vecFeatures.push_back(sumArray[index]);
    }

    GetLBPVec(imag, vecFeatures);

    delete [] sumArray;

    return vecFeatures;

}

void KUtility::TrainProcess(std::string dirName, std::string modelFileName, cv::Ptr<cv::ml::SVM> &svm)
{
    vector<string> imgLists;
    GetImgListByDir(dirName,imgLists);
    std::vector<int> vecLabels;
    std::vector<std::vector<float>> vecFeatures;
    std::string data_file("D:\\code\\train_data.txt");

    std::fstream fs(data_file.c_str(),std::ios_base::out | std::ios_base::trunc);

    fs<<"";
    fs.close();

    for(auto fileName : imgLists){
        ExtractFeatures(fileName, data_file, vecLabels, vecFeatures);
    }

    if(vecFeatures.empty() || vecLabels.empty()) return ;

    cv::Mat trainingDataMat(static_cast<int>(vecFeatures.size()), static_cast<int>((vecFeatures.begin())->size()), CV_32FC1, vecFeatures.data());
    cv::Mat labelsMat(static_cast<int>(vecLabels.size()), 1, CV_32SC1, vecLabels.data());
    //trainingDataMat.convertTo(trainingDataMat, CV_32FC1);
    //labelsMat.convertTo(labelsMat,CV_32SC1);


//    svm->setType(cv::ml::SVM::C_SVC);
//    svm->setKernel(cv::ml::SVM::RBF);
//    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
//    svm->setC(512.0f);
//    svm->setGamma(0.0078125f);
//    svm->train(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);
//    svm->save(modelFileName);

    BuildModelAuto(svm, modelFileName, labelsMat, trainingDataMat);
}

void KUtility::PredictProcess(std::string trainDirName, std::string modelFileName, std::string predictDirName)
{
    cv::Ptr<cv::ml::SVM> svm;
    try{
         svm = cv::ml::StatModel::load<cv::ml::SVM>(modelFileName);
    }catch(cv::Exception err){
        svm = cv::ml::SVM::create();
        TrainProcess(trainDirName, modelFileName, svm);
    }
    vector<string> imgLists;
    GetImgListByDir(predictDirName,imgLists);
    std::vector<std::vector<float>> vecFeatures;
    std::string data_file("D:\\code\\predict_data.txt");

    std::fstream fs(data_file.c_str(),std::ios_base::out | std::ios_base::trunc);

    fs<<"";
    fs.close();

    for(auto fileName : imgLists){
        ExtractFeatures(fileName, data_file, vecFeatures);
    }

    if(vecFeatures.empty()) return ;

    cv::Mat predictDataMat(static_cast<int>(vecFeatures.size()), static_cast<int>((vecFeatures.begin())->size()), CV_32FC1, vecFeatures.data());
    cv::Mat labelsMat(static_cast<int>(vecFeatures.size()), 1, CV_32SC1);

    //cv::Mat predictDataMat(vecFeatures);
    //cv::Mat labelsMat(vecLabels);
    //predictDataMat.convertTo(predictDataMat, CV_32FC1);
    //labelsMat.convertTo(labelsMat,CV_32SC1);


    svm->predict(predictDataMat, labelsMat);

    const unsigned char * da = labelsMat.ptr();
    for(int inde =0;inde < labelsMat.cols*labelsMat.rows;++inde){
        std::cout<<da[inde]+"0"<<" ";
    }
    return;
}

void KUtility::BuildModelAuto(cv::Ptr<cv::ml::SVM>& autoTrainSVM, std::string modelName, cv::Mat labelMat, cv::Mat featureMat)
{
    cv::Ptr<cv::ml::TrainData> trainDataArray = cv::ml::TrainData::create(featureMat, cv::ml::ROW_SAMPLE, labelMat);

    autoTrainSVM->setType(cv::ml::SVM::C_SVC);
    autoTrainSVM->setKernel(cv::ml::SVM::RBF);
    autoTrainSVM->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

    //autoTrainSVM->setC(512.0f);
    //autoTrainSVM->setGamma(0.0078125f);

    cv::ml::ParamGrid donotCalcParam(0, 0, 0);
    autoTrainSVM->trainAuto(trainDataArray, 10, cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C),
                            cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA), donotCalcParam, donotCalcParam, donotCalcParam, donotCalcParam);

    //float a=autoTrainSVM->getGamma();
    autoTrainSVM->save(modelName);
}

void KUtility::ExtractFeatures(std::string file, std::string vecFileName, std::vector<std::vector<float>>& featureVec)
{
    cv::Mat srcMat = cv::imread(file.c_str());

    cv::cvtColor(srcMat,srcMat,CV_BGR2GRAY);

    double thres=KUtility::ThresOTSU(srcMat);

    cv::threshold(srcMat,srcMat,thres,255,cv::THRESH_BINARY);

    vector<int> tempCol;
    vector<int> tempRow;

    RemoveSmall(srcMat);

    if(-1 != KUtility::SplitNumber(srcMat,tempRow,tempCol))
    {
        if( 2 != tempCol.size()) return;

        std::fstream fs(vecFileName.c_str(),std::ios_base::out | std::ios_base::app);

        int up = *(tempCol.begin()) + 1;
        int below = *(tempCol.rbegin());
        for(auto it = tempRow.begin(); it != tempRow.end(); ++it){
            int startCol = *(it++) + 1;
            int endCol = *it;
            cv::Mat numROI = srcMat(cv::Range(up,below),cv::Range(startCol,endCol));
           // cv::imshow("gray",numROI);
            std::vector<float> tempVec = GetFeatrueVec(numROI);
            featureVec.push_back(tempVec);
            for(auto itemFeature : tempVec){
                std::ostringstream float2str("");
                float2str<<itemFeature;
                fs<<float2str.str()<<" ";
            }

            fs<<"\r\n";
        }

        fs.close();
    }
}

void KUtility::RemoveSmall(cv::Mat _src, int small_thres)
{
    //static int imgcnt = 0;

    int width = _src.cols;
    int height = _src.rows;
    const int MAX_LABELS = width * height;
    int labelCount = 0;
    char xDelta[4] = {-1, -1, 0, 1};
    char yDelta[4] = {0, -1, -1, -1};

    int *labelMap = new(std::nothrow) int[MAX_LABELS];
    if(nullptr == labelMap) return ;

    int *labelArray = new(std::nothrow) int[width * height];
    if(nullptr == labelArray){ delete [] labelMap; return ; }

    memset(labelArray, 0, width * height * sizeof(int));

    labelMap[0] = 0;

    uchar* img = _src.ptr();
    // use eight neighbor
    for(int iY = 0;iY < height;++iY){
        for(int iX = 0;iX < width;++iX){
            if(img[iY * width + iX] > 128) continue;
            int minLabel = std::numeric_limits<int>::max();
            std::vector<int> vecLabel;

            for(int index = 0;index < 4;++index){
                int tempX = iX + xDelta[index];
                int tempY = iY + yDelta[index];
                // insure a valid position
                if(tempX < 0 || tempY < 0 || tempX >= width) continue;

                if(labelArray[tempY * width + tempX] != 0){
                    int mappedLabel = labelMap[labelArray[tempY * width + tempX]];
                    vecLabel.push_back(labelArray[tempY * width + tempX]);
                    if(mappedLabel < minLabel) minLabel = mappedLabel;
                }
            }

            if(vecLabel.empty()){
                labelArray[iY * width + iX] = ++labelCount;
                labelMap[labelCount] = labelCount;
                if(labelCount > MAX_LABELS) break;
            }else labelArray[iY * width + iX] = minLabel;
            for(auto tempLabel : vecLabel){
                labelMap[labelMap[tempLabel]] = minLabel;
                labelMap[tempLabel] = minLabel;
            }
        }
        if(labelCount > MAX_LABELS) break;
    }

    int *numLabels = new(std::nothrow) int[labelCount + 1];
    if(nullptr == numLabels){ delete [] labelMap; delete [] labelArray; return ; }

    memset(numLabels, 0, (labelCount + 1) * sizeof(int));
    for(int index = 0;index < width * height;++index){
        //labelArray[index] =labelMap[labelArray[index]] ==12?55000:0;//
        labelArray[index] = labelMap[labelArray[index]];
        numLabels[labelArray[index]]++;
    }

    for(int indexLable = 0;indexLable < labelCount + 1;++indexLable){
        if(numLabels[indexLable] > 0 && numLabels[indexLable] <= small_thres){
            for(int index = 0;index < width * height;++index){
                if(indexLable == labelArray[index]){
                    img[index] = 255;
                    labelArray[index] = 0;
                }
            }
        }
    }
//    for(int index = 0;index < width * height;++index){

//        labelArray[index] = 8000*labelArray[index];

//    }

//    cv::Mat mm(height, width, CV_32SC1, labelArray);

//cv::namedWindow("testw");
//cv::imshow("testw",mm);
//cvWaitKey(0);
//    cv::imwrite("C:\\test.jpg",mm);
    delete [] labelArray;
    delete [] labelMap;
    delete [] numLabels;
}

void KUtility::CreatFileUseLIBSVM(std::string trainDirName, std::string PredictDirName)
{
    vector<string> imgLists;
    GetImgListByDir(trainDirName,imgLists);
    std::string train_data_file("D:\\code\\train_data_LIBSVM.txt");

    std::fstream train_fs(train_data_file.c_str(),std::ios_base::out | std::ios_base::trunc);

    train_fs<<"";
    train_fs.close();

    for(auto fileName : imgLists){
        ExtractFeaturesLIBSVM(fileName, train_data_file);
    }

    imgLists.clear();

    GetImgListByDir(PredictDirName,imgLists);
    std::string predict_data_file("D:\\code\\predict_data_LIBSVM.txt");

    std::fstream predict_fs(predict_data_file.c_str(),std::ios_base::out | std::ios_base::trunc);

    predict_fs<<"";
    predict_fs.close();

    for(auto fileName : imgLists){
        ExtractFeaturesLIBSVM(fileName, predict_data_file);
    }
}

void KUtility::GetLBPVec(cv::Mat imag, std::vector<float> &vecLBP)
{
    int width = imag.cols;
    int height = imag.rows;

    char xDelta[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    char yDelta[8] = {0, -1, -1, -1, 0, 1, 1, 1};

    int *lbpArray = new(std::nothrow) int[width * height];
    if(nullptr == lbpArray) return ;

    memset(lbpArray, 0, width * height * sizeof(int));

    const uchar* src = imag.ptr();

    for(int iY = 0;iY < height;++iY){
        for(int iX = 0;iX < width;++iX){
            int blackCount = 0;
            for(int index = 0;index < 8;++index){
                int tempX = iX + xDelta[index];
                int tempY = iY + yDelta[index];
                // insure a valid position
                if(tempX < 0 || tempY < 0 || tempX >= width || tempY >= height) continue;
                if(src[tempY * width + tempX] < 128) blackCount++;
            }
            lbpArray[iY * width + iX] = blackCount;
        }
    }

    float *lbpHist = new(std::nothrow) float[8];
    if(nullptr == lbpHist){ delete [] lbpArray; return ; }

    memset(lbpHist, 0, 8 * sizeof(float));

    for(int iY = 0;iY < height;++iY){
        for(int iX = 0;iX < width;++iX){
            lbpHist[lbpArray[iY * width + iX]]++;
        }
    }

    for(int index = 0;index < 8;++index){
        //lbpHist[index] /= (width * height);
        vecLBP.push_back(lbpHist[index]);
    }

    delete [] lbpHist;
    delete [] lbpArray;

}

void KUtility::FeatureScale(std::string fileName, std::string scalParam)
{
    std::fstream scale_fs(scalParam.c_str(),std::ios_base::in);
    string tempStr("");
    int dim = 0;

    getline(scale_fs, tempStr, '\n');
    std::istringstream dim_strbuff(tempStr);
    dim_strbuff >> dim;

    float *maxArray = new(std::nothrow) float[dim];
    if(nullptr == maxArray){ scale_fs.close(); return ; }

    float *minArray = new(std::nothrow) float[dim];
    if(nullptr == minArray) { delete [] maxArray; scale_fs.close(); return ; }

    float lower = 0.f;
    float upper = 0.f;
    getline(scale_fs, tempStr, '\n');
    std::istringstream bound_strbuff(tempStr);
    bound_strbuff >> lower >> upper;
    int dimsCount = 0;
    while(getline(scale_fs, tempStr, '\n')){
        std::istringstream strbuff(tempStr);
        strbuff>>maxArray[dimsCount]>>minArray[dimsCount];
        ++dimsCount;
    }
    scale_fs.close();
//    for(int index = 0;index < dim;++index){
//        if(std::fabs(maxArray[index] - minArray[index]) < FLT_EPSILON ) continue;
//        maxArray[index] -= minArray[index];
//    }
    std::fstream scale_in(fileName.c_str(),std::ios_base::in);
    std::fstream scale_out((fileName+".scale").c_str(),std::ios_base::out | std::ios_base::trunc);

    while(getline(scale_in, tempStr, '\n')){
        std::istringstream strInBuff(tempStr);
        std::ostringstream strOutBuff;
        float tempValue = 0.f;
        dimsCount = 0;
        while(strInBuff>>tempValue){
            if(std::fabs(maxArray[dimsCount] - minArray[dimsCount]) < FLT_EPSILON){
                tempValue = (lower + upper)/2.f;
            }else{
                if(std::fabs(tempValue - minArray[dimsCount]) < FLT_EPSILON)
                    tempValue = lower;
                else if(std::fabs(tempValue - maxArray[dimsCount]) < FLT_EPSILON)
                    tempValue = upper;
                else{
                    tempValue = lower + (upper-lower) *
                        (tempValue-minArray[dimsCount])/
                        (maxArray[dimsCount]-minArray[dimsCount]);
                }
            }

            strOutBuff << tempValue << " ";
            ++dimsCount;
        }
        strOutBuff << "\r\n";
        scale_out << strOutBuff.str();
    }

    scale_in.close();
    scale_out.close();

    delete [] maxArray;
    delete [] minArray;
}

int KUtility::GetFeatureDims(std::string fileName)
{
    std::fstream data_fs(fileName.c_str(),std::ios_base::in);
    string tempStr("");
    int dim = 0;
    while(getline(data_fs, tempStr, '\n')){
        std::istringstream strbuff(tempStr);
        float temp;
        int dimsCount = 0;
        while(strbuff>>temp) ++dimsCount;
        if(dim && dimsCount != dim){
            data_fs.close();
            return -1;
        }
        dim = dimsCount;
    }

    data_fs.close();
    return dim;
}

void KUtility::CreateScaleParam(std::string fileName, std::string paramFile, float lower, float upper)
{
    std::fstream data_fs(fileName.c_str(),std::ios_base::in);

    string tempStr("");
    int dim = GetFeatureDims(fileName);
    if(-1 == dim || upper <= lower){ data_fs.close(); return ; }

    float *maxArray = new(std::nothrow) float[dim];
    if(nullptr == maxArray){ data_fs.close(); return ; }

    float *minArray = new(std::nothrow) float[dim];
    if(nullptr == minArray){ delete [] maxArray; data_fs.close(); return ; }

    float min_limit = std::numeric_limits<float>::min();
    float max_limit = std::numeric_limits<float>::max();

//    memset(maxArray, min_limit, dim * sizeof(float));
//    memset(minArray, max_limit, dim * sizeof(float));

    for(int index = 0;index < dim;++index){
        maxArray[index] = min_limit;
        minArray[index] = max_limit;
    }

    while(getline(data_fs, tempStr, '\n')){
        int dimsCount = 0;
        std::istringstream strbuff(tempStr);
        float temp;
        while(strbuff >> temp){
            if(maxArray[dimsCount] < temp){
                maxArray[dimsCount] = temp;
            }
            if(minArray[dimsCount] > temp){
                minArray[dimsCount] = temp;
            }
            ++dimsCount;
        }
    }

//    for(int index = 0;index < dim;++index){
//        if(std::fabs(maxArray[index] - minArray[index]) < FLT_EPSILON ) continue;
//        maxArray[index] -= minArray[index];
//    }

    std::fstream param_fs(paramFile.c_str(),std::ios_base::out | std::ios_base::trunc);
    std::ostringstream strbuff;
    strbuff << dim << "\r\n";
    strbuff << lower << " " << upper << "\r\n";
    for(int index = 0;index < dim;++index){
        strbuff << maxArray[index] << " " << minArray[index] << "\r\n";
    }
    param_fs << strbuff.str();
    param_fs.close();

    delete [] maxArray;
    delete [] minArray;

    data_fs.close();
}

void KUtility::ExtractFeaturesLIBSVM(std::string file, std::string vecFileName)
{
    cv::Mat srcMat = cv::imread(file.c_str());

    cv::cvtColor(srcMat,srcMat,CV_BGR2GRAY);

    double thres=KUtility::ThresOTSU(srcMat);

    cv::threshold(srcMat,srcMat,thres,255,cv::THRESH_BINARY);
    vector<int> tempCol;
    vector<int> tempRow;

    char fname[_MAX_FNAME];
    _splitpath(file.c_str(), NULL, NULL, fname, NULL);
    string fileNameExt(fname);

    RemoveSmall(srcMat);

    if(-1 != KUtility::SplitNumber(srcMat,tempRow,tempCol))
    {
        if( 2 != tempCol.size()) return;

        if(fileNameExt.length() != tempRow.size() / 2) return;

        std::fstream fs(vecFileName.c_str(),std::ios_base::out | std::ios_base::app);

        int up = *(tempCol.begin()) + 1;
        int below = *(tempCol.rbegin());
        int col_count = 0;

        for(auto it = tempRow.begin(); it != tempRow.end(); ++it){
            int startCol = *(it++) + 1;
            int endCol = *it;
            cv::Mat numROI = srcMat(cv::Range(up,below),cv::Range(startCol,endCol));
            std::vector<float> tempVec = GetFeatrueVec(numROI);
            fs<<fileNameExt.at(col_count++);
            fs<<" ";
            int vecCount = 1;
            for(auto itemFeature : tempVec){
                std::ostringstream float2str("");
                float2str<<vecCount++;
                float2str<<":";
                float2str<<itemFeature;
                fs<<float2str.str()<<" ";
            }

            fs<<"\r\n";
        }

        fs.close();
    }
}
