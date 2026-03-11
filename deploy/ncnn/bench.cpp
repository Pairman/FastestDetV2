#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "allocator.h"
#include "benchmark.h"
#include "cpu.h"
#include "net.h"

static double percentile(const std::vector<double>& sorted_v, double p)
{
	if (sorted_v.empty())
		return 0.0;
	const double rank = (p / 100.0) * (sorted_v.size() - 1);
	const int lo = (int)rank;
	const int hi = std::min((int)sorted_v.size() - 1, lo + 1);
	const double w = rank - lo;
	return sorted_v[lo] * (1.0 - w) + sorted_v[hi] * w;
}

// https://github.com/RangiLyu/nanodet/blob/be9b4a9/demo_ncnn/main.cpp#L293
int main(int argc, char **argv)
{
	const char *param = "fastestdetv2.param";
	const char *bin = "fastestdetv2.bin";
	const char *input_blob = "in0";
	const char *output_blob = "out0";
	int iters = 300;
	int threads = 4;
	int w = 352;
	int h = 352;
	// ./bench [ITERS] [THREADS]
	if (argc > 1)
		iters = std::atoi(argv[1]);
	if (argc > 2)
		threads = std::atoi(argv[2]);

	ncnn::set_cpu_powersave(2);
	ncnn::Net net;
	net.opt.num_threads = threads;
	if (net.load_param(param) != 0) {
		std::printf("Failed to load param: %s\n", param);
		return -1;
	}
	if (net.load_model(bin) != 0) {
		std::printf("Failed to load bin: %s\n", bin);
		return -1;
	}

	ncnn::Mat input(w, h, 3);
	input.fill(0.01f);
	const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
	input.substract_mean_normalize(0, norm_vals);
	int warmup = (iters > 10) ? 10 : (iters / 2);
	for (int i = 0; i < warmup; ++i) {
		ncnn::Extractor ex = net.create_extractor();
		ex.input(input_blob, input);

		ncnn::Mat out;
		if (ex.extract(output_blob, out) != 0) {
			std::printf("Failed to extract %s during warmup\n", output_blob);
			return -1;
		}
	}

	std::vector<double> times;
	times.reserve(iters);
	double tmin = DBL_MAX;
	double tmax = -DBL_MAX;
	double tsum = 0.0;
	for (int i = 0; i < iters; ++i) {
		const double t0 = ncnn::get_current_time();
		ncnn::Extractor ex = net.create_extractor();
		ex.input(input_blob, input);
		ncnn::Mat out;
		if (ex.extract(output_blob, out) != 0) {
			std::printf("Failed to extract %s at iter %d\n", output_blob, i);
			return -1;
		}
		const double t1 = ncnn::get_current_time();
		const double dt = t1 - t0;
		times.push_back(dt);
		tmin = (std::min)(tmin, dt);
		tmax = (std::max)(tmax, dt);
		tsum += dt;
	}

	std::sort(times.begin(), times.end());
	const double avg = tsum / (double)times.size();
	const double p95 = percentile(times, 95.0);
	const double p99 = percentile(times, 99.0);
    std::printf("Model: %s + %s\n", param, bin);
    std::printf("Input: %s  Output: %s  Shape: 1x3x%dx%d\n", input_blob, output_blob, h, w);
    std::printf("Warmup: %d  Iters: %d  Threads: %d\n", warmup, iters, threads);
    std::printf("Latency(ms): min=%.3f  max=%.3f  avg=%.3f  p95=%.3f  p99=%.3f\n",
        tmin, tmax, avg, p95, p99);
	return 0;
}
