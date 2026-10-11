"""Check single-node CUDA collectives before loading the diffusion models."""
import argparse
import ctypes
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def nccl_versions():
    # torch.cuda.nccl.version() reports compile-time headers in PyTorch 2.0.
    libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                        if "libnccl.so" in line and line.split()[-1].startswith("/")})
    runtime = {}
    for path in libraries:
        library = ctypes.CDLL(path)
        version = ctypes.c_int()
        library.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.ncclGetVersion.restype = ctypes.c_int
        code = library.ncclGetVersion(ctypes.byref(version))
        if code != 0:
            raise RuntimeError("ncclGetVersion failed: %s" % code)
        runtime[path] = version.value
    return {"nccl_compiled": torch.cuda.nccl.version(), "nccl_runtime": runtime}


def worker(rank, devices, port, output):
    torch.cuda.set_device(rank)
    record = {"rank": rank, "gpu": torch.cuda.get_device_name(rank),
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "checks": [], **nccl_versions()}
    try:
        dist.init_process_group("nccl", init_method="tcp://127.0.0.1:%d" % port,
                                rank=rank, world_size=devices,
                                timeout=timedelta(seconds=60))
        dist.barrier(device_ids=[rank])
        for count in (1, 1048576, 16777216):
            value = torch.full((count,), float(rank + 1), device="cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            dist.all_reduce(value)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            assert bool(torch.all(value == devices * (devices + 1) / 2)), "all_reduce mismatch"
            value.fill_(7.0 if rank == 0 else 0.0)
            dist.broadcast(value, src=0)
            assert bool(torch.all(value == 7.0)), "broadcast mismatch"
            record["checks"].append({"float32_elements": count, "all_reduce_seconds": seconds})
        dist.barrier(device_ids=[rank])
        record["status"] = "passed"
        print(json.dumps(record), flush=True)
    except Exception as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        (Path(output) / ("rank%d.json" % rank)).write_text(json.dumps(record, indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")  # Runner keeps the experiment configuration snapshot.
    parser.add_argument("--output", required=True)
    parser.add_argument("--devices", type=int, default=2)
    args = parser.parse_args()
    if args.devices < 2 or torch.cuda.device_count() < args.devices:
        parser.error("At least two visible allocated GPUs are required")
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {"torch": torch.__version__, "cuda": torch.version.cuda,
                "hostname": socket.gethostname(), **nccl_versions(),
                "environment": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")
                                or k in ("LD_PRELOAD", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES")}}
    (directory / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(args.devices, port, str(directory)), nprocs=args.devices, join=True)
    print("NCCL barrier, all_reduce and broadcast passed on all ranks", flush=True)


if __name__ == "__main__":
    main()
