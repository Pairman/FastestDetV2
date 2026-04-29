from argparse import ArgumentParser
from pathlib import Path
from rknn.api import RKNN

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True, help=".onnx model path")
    parser.add_argument("--target", type=str, default="rk3588", help="target platform")
    opt = parser.parse_args()
    outdir = Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(opt.onnx).resolve()
    rknn_path = outdir / f"{onnx_path.stem}.rknn"
    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(target_platform=opt.target, mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])
        if ret != 0:
            raise RuntimeError(f"rknn.config failed with code {ret}")
        ret = rknn.load_onnx(model=str(onnx_path), input_size_list=[[1, 3, 352, 352]])
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed with code {ret}")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed with code {ret}")
        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed with code {ret}")
    finally:
        rknn.release()
    print(f"Saved RKNN to {rknn_path}")
