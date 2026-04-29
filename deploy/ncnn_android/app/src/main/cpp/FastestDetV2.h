#ifndef FASTESTDETV2_H
#define FASTESTDETV2_H

#include "net.h"
#include <android/bitmap.h>
#include <android/asset_manager.h>

struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
};

class FastestDetV2 {
public:
    FastestDetV2(AAssetManager *mgr, const char *param, const char *bin);

    ~FastestDetV2();

    bool isLoaded() const;
    std::vector<BoxInfo> detect(JNIEnv *env, jobject image, float score_threshold, float nms_threshold);
    std::string bench(int iters);
private:
    void preprocess(JNIEnv *env, jobject image, ncnn::Mat& in);
    static std::vector<BoxInfo> nms(std::vector<BoxInfo> boxes, float iou_threshold);
    static float iou(const BoxInfo& a, const BoxInfo& b);

    ncnn::Net *Net;
    bool loaded = false;
    int input_size[2] = {352, 352};
    int num_class = 80;


public:
    static FastestDetV2 *detector;
};


#endif // FASTESTDETV2_H
