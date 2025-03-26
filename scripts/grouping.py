import triton
import torch
import triton.language as tl
import random



@triton.autotune(
    configs=[

        # triton.Config({"Br": 128, "Bc": 256, "Bk": 32, "GROUP_SZE_M": 4}, num_stages=3, num_warps=8),
        # triton.Config({"Br": 16, "Bc": 16, "Bk": 16, "GROUP_SZE_M": 1}, num_stages=3, num_warps=8),

        triton.Config({"Br": 64, "Bc": 32, "Bk": 64, "GROUP_SZE_M": 8}, num_stages=4, num_warps=8),
        # triton.Config({"Br": 128, "Bc": 128, "Bk": 32, "GROUP_SZE_M": 4}, num_stages=5, num_warps=8),
        # triton.Config({"Br": 128, "Bc": 128, "Bk": 64, "GROUP_SZE_M": 2}, num_stages=4, num_warps=8),
  
    ],
    key=["M", "N", "K"],
)

@triton.jit
def grouping(A, B, C,
                am_stride, ak_stride,
                bk_stride, bn_stride,
                cm_stride, cn_stride,
                M, N, K,
                Br:tl.constexpr, Bc:tl.constexpr, Bk:tl.constexpr,
                GROUP_SZE_M:tl.constexpr,
                ):
    # import pdb; pdb.set_trace()
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, Br)
    num_pid_n = tl.cdiv(N, Bc)
    num_pid_in_group = GROUP_SZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * Br + tl.arange(0, Br)) % M
    offs_bn = (pid_n * Bc + tl.arange(0, Bc)) % N

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, Br), Br)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, Bc), Bc)

    offs_k = tl.arange(0, Bk)
    a_ptrs = A + (offs_am[:, None] * am_stride + offs_k[None, :] * ak_stride)
    b_ptrs = B + (offs_k[:, None] * bn_stride  + offs_bn[None, :] * bk_stride)

    accumulator = tl.zeros((Br, Bc), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, Bk)):

        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * Bk, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * Bk, other=0.0)

        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += Bk * ak_stride
        b_ptrs += Bk * bn_stride 

    
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * Br + tl.arange(0, Br)
    offs_cn = pid_n * Bc + tl.arange(0, Bc)
    c_ptrs = C + cm_stride * offs_cm[:, None] + cn_stride * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)



def kernel(A, B, C):
    M, N = C.shape
    M, K = A.shape

    grid = lambda META: (triton.cdiv(M, META['Br']) * triton.cdiv(N, META['Bc']), 1)
    
    grouping[grid](A, B, C,
                                                   A.stride(0), A.stride(1),
                                                   B.stride(0), B.stride(1),
                                                   C.stride(0), C.stride(1),
                                                   M, N, K)
    return C


def normal_impl(A, B):

    return torch.matmul(A,B)

def test():
    torch.manual_seed(20)

    M = K = N = 4096
    device = 'cuda:0'
    dtype = torch.float16
    A = torch.randn(size = (M, K), device=device, dtype=dtype)
    B = torch.randn(size = (K, N), device=device, dtype=dtype)
    C = torch.randn(size = (M, N), device=device, dtype=dtype)
    random.seed(20)

    ref_out = normal_impl(A, B)
    B = B.T.contiguous()
    tr_out = kernel(A, B, C)
   


    print(f"correct: {torch.allclose(ref_out, tr_out, atol = 1e-2, rtol=0)}")
    print(f"dist: {torch.dist(ref_out, tr_out):.10f} | max dist : {(torch.max(torch.abs(ref_out - tr_out))):.10f}")

def run():

    M = K = N = 4096
    device = 'cuda:0'
    dtype = torch.float16
    A = torch.randn(size = (M, K), device=device, dtype=dtype)
    B = torch.randn(size = (K, N), device=device, dtype=dtype)
    C = torch.randn(size = (M, N), device=device, dtype=dtype)
    random.seed(20)

    tr_out = kernel(A, B, C)


@triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[(2**i) for i in range(12)],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names= ['Triton', 'Torch'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='GB/s',
            plot_name='perf',
            args = {}
    
        )
)

def benchmark(N, provider):
    torch.manual_seed(20)

    M = K = N 
    device = torch.device('cuda:0')
    dtype = torch.float16
    A = torch.randn(size = (M, K), device=device, dtype=dtype)
    B = torch.randn(size = (K, N), device=device, dtype=dtype)
    C = torch.randn(size = (M, N), device=device, dtype=dtype)
    
    stream = getattr(torch, device.type).Stream()
    getattr(torch, device.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: normal_impl(A,B))
    
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: kernel(A,B,C))
         

    gbps = lambda ms: ((2*N**3 + N**2)/1e9)/ (ms*1e-3)

    return gbps(ms)

if __name__ == "__main__":

    # benchmark.run(show_plots=True, print_data=True)
    run()