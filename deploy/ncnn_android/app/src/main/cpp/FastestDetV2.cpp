#include "FastestDetV2.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <sstream>
#include <vector>

#include "benchmark.h"
#include "cpu.h"

FastestDetV2* FastestDetV2::detector = nullptr;

namespace {
struct BenchStats {
    double min_ms = DBL_MAX;
    double max_ms = -DBL_MAX;
    double avg_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
};

static double percentile(const std::vector<double>& sorted_v, double p)
{
    if (sorted_v.empty()) {
        return 0.0;
    }
    const double rank = (p / 100.0) * (sorted_v.size() - 1);
    const int lo = (int)rank;
    const int hi = std::min((int)sorted_v.size() - 1, lo + 1);
    const double w = rank - lo;
    return sorted_v[lo] * (1.0 - w) + sorted_v[hi] * w;
}

static float sigmoid(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

static BenchStats run_bench(ncnn::Net* net, const int input_h, const int input_w, int threads, int iters)
{
    ncnn::set_cpu_powersave(2);
    net->opt.num_threads = threads;

    ncnn::Mat input(input_w, input_h, 3);
    input.fill(0.01f);
    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    input.substract_mean_normalize(0, norm_vals);

    const int warmup = (iters > 10) ? 10 : (iters / 2);
    for (int i = 0; i < warmup; ++i) {
        ncnn::Extractor ex = net->create_extractor();
        ex.set_light_mode(true);
        ex.input("in0", input);
        ncnn::Mat out;
        ex.extract("out0", out);
    }

    std::vector<double> times;
    times.reserve(iters);
    double sum = 0.0;
    BenchStats stats;
    for (int i = 0; i < iters; ++i) {
        const double t0 = ncnn::get_current_time();
        ncnn::Extractor ex = net->create_extractor();
        ex.set_light_mode(true);
        ex.input("in0", input);
        ncnn::Mat out;
        ex.extract("out0", out);
        const double dt = ncnn::get_current_time() - t0;
        times.push_back(dt);
        stats.min_ms = (std::min)(stats.min_ms, dt);
        stats.max_ms = (std::max)(stats.max_ms, dt);
        sum += dt;
    }

    if (!times.empty()) {
        std::sort(times.begin(), times.end());
        stats.avg_ms = sum / (double)times.size();
        stats.p95_ms = percentile(times, 95.0);
        stats.p99_ms = percentile(times, 99.0);
    } else {
        stats.min_ms = 0.0;
        stats.max_ms = 0.0;
    }
    return stats;
}
}

FastestDetV2::FastestDetV2(AAssetManager *mgr, const char *param, const char *bin)
{
    this->Net = new ncnn::Net();
    this->Net->opt.use_fp16_arithmetic = true;
    this->Net->opt.use_fp16_packed = true;
    this->Net->opt.use_fp16_storage = true;
    const int ret_param = this->Net->load_param(mgr, param);
    const int ret_model = this->Net->load_model(mgr, bin);
    this->loaded = (ret_param == 0 && ret_model == 0);
}

FastestDetV2::~FastestDetV2()
{
    delete this->Net;
}

bool FastestDetV2::isLoaded() const
{
    return this->loaded;
}

void FastestDetV2::preprocess(JNIEnv *env, jobject image, ncnn::Mat& in)
{
    in = ncnn::Mat::from_android_bitmap_resize(
        env, image, ncnn::Mat::PIXEL_RGBA2BGR, input_size[1], input_size[0]);
    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    in.substract_mean_normalize(0, norm_vals);
}

float FastestDetV2::iou(const BoxInfo& a, const BoxInfo& b)
{
    const float xx1 = std::max(a.x1, b.x1);
    const float yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2);
    const float yy2 = std::min(a.y2, b.y2);
    const float w = std::max(0.f, xx2 - xx1);
    const float h = std::max(0.f, yy2 - yy1);
    const float inter = w * h;
    const float area_a = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    const float area_b = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    const float uni = area_a + area_b - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

std::vector<BoxInfo> FastestDetV2::nms(std::vector<BoxInfo> boxes, float iou_threshold)
{
    std::sort(boxes.begin(), boxes.end(), [](const BoxInfo& a, const BoxInfo& b) {
        return a.score > b.score;
    });

    std::vector<BoxInfo> keep;
    std::vector<char> removed(boxes.size(), 0);
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (removed[i]) {
            continue;
        }
        keep.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (removed[j]) {
                continue;
            }
            if (boxes[i].label != boxes[j].label) {
                continue;
            }
            if (iou(boxes[i], boxes[j]) > iou_threshold) {
                removed[j] = 1;
            }
        }
    }
    return keep;
}

std::vector<BoxInfo> FastestDetV2::detect(JNIEnv *env, jobject image, float score_threshold, float nms_threshold)
{
    if (!this->loaded) {
        return {};
    }

    AndroidBitmapInfo img_size;
    if (AndroidBitmap_getInfo(env, image, &img_size) != ANDROID_BITMAP_RESULT_SUCCESS) {
        return {};
    }

    ncnn::Mat input;
    this->preprocess(env, image, input);
    if (input.empty()) {
        return {};
    }

    this->Net->opt.num_threads = 4;
    ncnn::Extractor ex = this->Net->create_extractor();
    ex.set_light_mode(true);
    ex.input("in0", input);

    ncnn::Mat out;
    ex.extract("out0", out);

    std::vector<BoxInfo> boxes;
    for (int y = 0; y < out.h; ++y) {
        for (int x = 0; x < out.w; ++x) {
            const float obj = out.channel(0).row(y)[x];
            int cls = 0;
            float cls_max = 0.f;
            for (int i = 0; i < num_class; ++i) {
                const float cls_score = out.channel(5 + i).row(y)[x];
                if (cls_score > cls_max) {
                    cls_max = cls_score;
                    cls = i;
                }
            }

            const float score = std::pow(obj, 0.6f) * std::pow(cls_max, 0.4f);
            if (score < score_threshold) {
                continue;
            }

            const float tx = std::tanh(out.channel(1).row(y)[x]);
            const float ty = std::tanh(out.channel(2).row(y)[x]);
            const float tw = sigmoid(out.channel(3).row(y)[x]);
            const float th = sigmoid(out.channel(4).row(y)[x]);
            const float cx = (x + tx) / out.w;
            const float cy = (y + ty) / out.h;

            BoxInfo box;
            box.x1 = (cx - 0.5f * tw) * img_size.width;
            box.y1 = (cy - 0.5f * th) * img_size.height;
            box.x2 = (cx + 0.5f * tw) * img_size.width;
            box.y2 = (cy + 0.5f * th) * img_size.height;
            box.x1 = std::max(0.f, std::min(box.x1, (float)(img_size.width - 1)));
            box.y1 = std::max(0.f, std::min(box.y1, (float)(img_size.height - 1)));
            box.x2 = std::max(0.f, std::min(box.x2, (float)(img_size.width - 1)));
            box.y2 = std::max(0.f, std::min(box.y2, (float)(img_size.height - 1)));
            if (box.x2 <= box.x1 || box.y2 <= box.y1) {
                continue;
            }
            box.label = cls;
            box.score = score;
            boxes.push_back(box);
        }
    }

    return nms(boxes, nms_threshold);
}

std::string FastestDetV2::bench(int iters)
{
    if (iters <= 0) {
        return "Invalid bench iteration count.";
    }
    if (!this->loaded) {
        return "FastestDetV2 model load failed.";
    }

    const BenchStats stats_1t = run_bench(this->Net, this->input_size[0], this->input_size[1], 1, iters);
    const BenchStats stats_4t = run_bench(this->Net, this->input_size[0], this->input_size[1], 4, iters);

    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    oss << "FastestDetV2 NCNN Bench\n";
    oss << "Shape: 1x3x" << this->input_size[0] << "x" << this->input_size[1] << "\n";
    oss << "Iters: " << iters << "  Warmup: " << ((iters > 10) ? 10 : (iters / 2)) << "\n";
    oss << "1x core: min=" << stats_1t.min_ms
        << "  max=" << stats_1t.max_ms
        << "  avg=" << stats_1t.avg_ms
        << "  p95=" << stats_1t.p95_ms
        << "  p99=" << stats_1t.p99_ms << "\n";
    oss << "4x core: min=" << stats_4t.min_ms
        << "  max=" << stats_4t.max_ms
        << "  avg=" << stats_4t.avg_ms
        << "  p95=" << stats_4t.p95_ms
        << "  p99=" << stats_4t.p99_ms;
    return oss.str();
}
