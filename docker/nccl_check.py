#!/usr/bin/env python3
"""Multi-GPU NCCL sanity check: all-reduce 256 MiB on every GPU, report bus bandwidth.
Run with: torchrun --standalone --nproc_per_node=N nccl_check.py
"""
import datetime, os, time
import torch
import torch.distributed as dist


def main():
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
    world, rank = dist.get_world_size(), dist.get_rank()
    x = torch.ones(64 * 1024 * 1024, device="cuda", dtype=torch.float32)  # 256 MiB
    for _ in range(3):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    x.fill_(1.0)
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    ok = torch.allclose(x, torch.full_like(x, float(world) ** iters))
    flag = torch.tensor([1 if ok else 0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        busbw = x.numel() * 4 * 2 * (world - 1) / world / dt / 1e9
        status = "PASS" if flag.item() == 1 else "FAIL"
        print(f"{status} NCCL all-reduce across {world} GPUs: {dt * 1e3:.1f} ms per 256 MiB, bus bandwidth ~{busbw:.0f} GB/s")
        if status == "PASS" and world > 1 and busbw < 20:
            print("WARN NCCL bandwidth is low (<20 GB/s): GPUs may be PCIe without NVLink; training will be slower.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
