import time, torch
torch.backends.cuda.matmul.allow_tf32 = True
n = 2048
for dev in ("cuda",):
    a = torch.randn(n, n, dtype=torch.float16, device=dev)
    b = torch.randn(n, n, dtype=torch.float16, device=dev)
    for _ in range(10): c = a @ b
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(100): c = a @ b
    torch.cuda.synchronize(); dt = (time.time() - t0) / 100
    print(f"{torch.cuda.get_device_name(0)}: {2*n**3/dt/1e12:.1f} TFLOP/s fp16  "
          f"peak_free_mem={torch.cuda.mem_get_info()[0]/2**30:.1f} GiB of "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
