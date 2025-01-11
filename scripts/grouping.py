import triton
import torch
import triton.language as tl
import random

@triton.jit
def grouping(A, B, C, alpha, beta,
                Br:tl.constexpr, Bc:tl.constexpr, Bk:tl.constexpr,
                GROUP_SZE_M:tl.constexpr,
                am_stride:tl.constexpr, ak_stride:tl.constexpr,
                bk_stride:tl.constexpr, bn_stride:tl.constexpr,
                cm_stride:tl.constexpr, cn_stride:tl.constexpr,
                M:tl.constexpr, N:tl.constexpr, K:tl.constexpr
                ):
    
    pid = tl.program_id(0)
    
    num_pids_m = tl.cdiv(M, Br)
    num_pids_n = tl.cdiv(N, Bc)

    pids_per_group = num_pids_n * GROUP_SZE_M
    group_id = pid//pids_per_group
    start_pid_m = group_id*GROUP_SZE_M

    group_size_m = min(GROUP_SZE_M, num_pids_m-start_pid_m)

    pid_m = start_pid_m + (pid%pids_per_group)%group_size_m
    pid_n = (pid%pids_per_group)//group_size_m

    off_m = tl.arange(0, Br)
    off_k = tl.arange(0, Bk)
    off_n = tl.arange(0, Bc)

    a_offset = off_m[:, None]*am_stride + off_k[None, :]*ak_stride
    b_offset = off_k[:, None]*bk_stride + off_n[None, :]*bn_stride
    
    a_start_pos = A + pid_m*Br*am_stride
    b_start_pos = B + pid_n*Bc*bn_stride
    a_ptrs = (a_start_pos + a_offset)
    b_ptrs = (b_start_pos + b_offset)

    
    acc = tl.zeros([Br, Bc], dtype = tl.float32)
    for i in range(0, tl.cdiv(K, Bk)):
        mask_a = (off_k[None, :]+ (i*Bk))< K 
        mask_b = (off_k[:, None] + (i*Bk))< K 

        a = tl.load(a_ptrs, mask=mask_a , other= 0.0)
        b = tl.load(b_ptrs, mask=mask_b , other= 0.0)

        acc = tl.dot(a, b, acc, allow_tf32=False)
       
        a_ptrs += Br*ak_stride
        b_ptrs += Bc*bk_stride
    
    cm_off = pid_m * Br + tl.arange(0, Br)
    cn_off = pid_n * Bc + tl.arange(0, Bc)
    
    c_offsets = cm_off[:, None]*cm_stride + cn_off[None, :]*cn_stride
    c_ptrs = (C + c_offsets)
    c = tl.load(c_ptrs)
    
    out = alpha*acc + beta*c
    cmask =(cm_off[:, None]< M) & (cn_off[None, :] < N)
    tl.store(c_ptrs, out, mask = cmask)




def kernel(A, B, C, alpha, beta):
    
    Br = Bk = Bc = 64
    GROUP_SIZE_M = 8
    M, N = C.shape
    M, K = A.shape

    grouping[(triton.cdiv(M,Br) * triton.cdiv(N, Bc), 1)](A, B, C, alpha, beta, 
                                                   Br, Bc, Bk, GROUP_SIZE_M,
                                                   A.stride(0), A.stride(1),
                                                   B.stride(0), B.stride(1),
                                                   C.stride(0), C.stride(1),
                                                   M, N, K, num_warps = 4, num_stages = 2)
    
    return C


def normal_impl(A, B, C, alpha, beta):
    
    acc = torch.matmul(A,B)
    fp = alpha*acc
    sp = beta*C
    
    return (fp + sp)

def test():
    torch.manual_seed(20)

    M = K = N = 4092
    device = 'cuda:0'
    dtype = torch.float32
    A = torch.randn(size = (M, K), device=device, dtype=dtype)
    B = torch.randn(size = (K, N), device=device, dtype=dtype)
    C = torch.randn(size = (M, N), device=device, dtype=dtype)
    random.seed(20)
    beta = random.random()
    alpha = random.random()

    ref_out = normal_impl(A, B, C, alpha, beta)

    tr_out = kernel(A, B, C, alpha, beta)


    print(f"correct: {torch.allclose(ref_out, tr_out, atol = 1e-2, rtol=0)}")
    print(f"dist: {torch.dist(ref_out, tr_out):.10f} | max dist : {(torch.max(torch.abs(ref_out - tr_out))):.10f}")

@triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[(32*i) for i in range(30, 51)],
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
    dtype = torch.float32
    A = torch.randn(size = (M, K), device=device, dtype=dtype)
    B = torch.randn(size = (K, N), device=device, dtype=dtype)
    C = torch.randn(size = (M, N), device=device, dtype=dtype)
    random.seed(20)
    beta = random.random()
    alpha = random.random()
    stream = getattr(torch, device.type).Stream()
    getattr(torch, device.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: normal_impl(A,B,C, alpha, beta))
    
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: kernel(A,B,C, alpha, beta))
         

    gbps = lambda ms: ((2*N**3 + N**2)/1e9)/ (ms*1e-3)

    return gbps(ms)

if __name__ == "__main__":
    benchmark.run(show_plots=False, print_data=True)

    test()