import argparse
import time
import numpy as np
from ais_bench.infer.interface import InferSession

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="fastestdetv2.om", help=".om weights")
    parser.add_argument("--iters", type=int, default=300, help="iterations")
    parser.add_argument("--device", type=str, default="npu:0", help="device. npu:0, etc.")
    opt = parser.parse_args()
    device_id = int(opt.device.split(":")[1])
    # load model
    session = InferSession(device_id=device_id, model_path=opt.weights)
    # preproc
    img = np.ones((1, 3, 352, 352), dtype=np.float32)
    # warmup
    for _ in range(10):
        session.infer([img])
    # benchmark
    times = []
    for _ in range(opt.iters):
        t0 = time.perf_counter()
        session.infer([img])
        times.append((time.perf_counter() - t0) * 1000.0)
    # report
    print(f"Model: {opt.weights}")
    print(f"Input shape: 1x3x{352}x{352}")
    print(f"Warmup: 10  Iters: {opt.iters}  Device: {opt.device}")
    print("Latency(ms): "
        f"min={min(times):.3f}  max={max(times):.3f}  avg={np.mean(times):.3f}  "
        f"p95={np.percentile(times, 95.0):.3f}  p99={np.percentile(times, 99.0):.3f}")
