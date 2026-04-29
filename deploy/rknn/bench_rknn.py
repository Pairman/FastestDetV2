import argparse
from pathlib import Path
import time
import numpy as np
from rknnlite.api import RKNNLite

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rknn", type=str, required=True, help=".rknn path")
    parser.add_argument("--iters", type=int, default=300, help="iterations")
    parser.add_argument("--npus", type=int, default=1, help="number of npu cores. 1, 2, or 3")
    opt = parser.parse_args()
    rknn_path = Path(opt.rknn).resolve()
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(str(rknn_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed with code {ret}")
    ret = rknn.init_runtime(core_mask={1: RKNNLite.NPU_CORE_0, 2: RKNNLite.NPU_CORE_0_1, 3: RKNNLite.NPU_CORE_0_1_2}[opt.npus])
    if ret != 0:
        raise RuntimeError(f"init_runtime failed with code {ret}")
    inputs = [np.ones((1, 352, 352, 3), dtype=np.uint8)]
    for _ in range(10):
        outputs = rknn.inference(inputs=inputs, data_format="nhwc")
        if outputs is None:
            raise RuntimeError("Warmup inference returned None")
    times = []
    for _ in range(opt.iters):
        t0 = time.perf_counter()
        outputs = rknn.inference(inputs=inputs, data_format="nhwc")
        dt = (time.perf_counter() - t0) * 1000.0
        if outputs is None:
            raise RuntimeError("Inference returned None")
        times.append(dt)
    rknn.release()
    print(f"Model: {rknn_path}")
    print(f"Input shape: 1x3x{352}x{352}")
    print(f"Warmup: {opt.warmup}  Iters: {opt.iters}  Cores: {opt.npus}")
    print("Latency(ms): "
        f"min={min(times):.3f}  max={max(times):.3f}  avg={np.mean(times):.3f}  "
        f"p95={np.percentile(times, 95.0):.3f}  p99={np.percentile(times, 99.0):.3f}")
