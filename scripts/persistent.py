import torch
import triton
import triton.language as tl
from triton.runtime import driver



@triton.jit
def persistent(A, B, C,
               am_stride,
               bk_stride,
               cm_stride,
               NUM_SM: tl.constexpr,
               Br: tl.constexpr, Bc: tl.constexpr,Bk: tl.constexpr,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               GROUP_SZE: tl.constexpr):
    
    tile_id = tl.program_id(0)
    M_BLOCKS = tl.cdiv(M, Br)
    N_BLOCKS = tl.cdiv(N, Bc)
    K_BLOCKS = tl.cdiv(K, Bk)
    ki = -1
    pid_per_group = N_BLOCKS*GROUP_SZE


    k_off = tl.arange(0, Bk)
    TOTAL_TILES = M_BLOCKS * N_BLOCKS
    TILES_PER_SM = tl.cdiv(TOTAL_TILES,NUM_SM)
    accumulator = tl.zeros([Br, Bc], dtype = tl.float32)

    
    pid_m, pid_n = 0, 0
    m_off = tl.arange(0, Br)
    n_off = tl.arange(0, Bc)


    for i in range(K_BLOCKS * TILES_PER_SM):
        ki = tl.where(ki == K_BLOCKS-1, 0, ki+1)
        
        if ki == 0:
            group_id = tile_id//pid_per_group
            start_m = group_id*GROUP_SZE

            group_size = min(GROUP_SZE, M_BLOCKS-start_m)

            pid_m = start_m + (tile_id%pid_per_group)%group_size
            pid_n = (tile_id%pid_per_group)//group_size
            tile_id += NUM_SM
        

            m_off = tl.arange(0, Br)
            n_off = tl.arange(0, Bc)

        off_a = pid_m*Br*am_stride + m_off[:, None]*am_stride + k_off[None, :]
        off_b = pid_n*Bc + k_off[:, None]*bk_stride + n_off[None, :]

        off_a += Bk*(i%K_BLOCKS)
        off_b += Bk*bk_stride*(i%K_BLOCKS)

        a = tl.load(A + off_a, mask = k_off[None, :] < K - ki*Bk, other=0)
        b = tl.load(B + off_b, mask = k_off[:, None] < K - ki*Bk, other=0)

        accumulator = tl.dot(a, b, accumulator, allow_tf32=False)


        if ki == K_BLOCKS-1:
            c = accumulator.to(tl.float16)
            cm_off = (pid_m*Br + m_off)*cm_stride
            cn_off = pid_n*Bc + n_off
            offset = cm_off[:,None] + cn_off[None, :]
            mask = (cm_off[:, None] < M*cm_stride) & (cn_off[None, :] < N)
            tl.store(C + offset, c, mask)

            accumulator = tl.zeros([Br, Bc], dtype = tl.float32)


def matmul_kernel(A, B, C, NUM_SM):
    M,K = A.shape
    K,N = B.shape
    
    Br = 64
    Bc = 128
    Bk = 64
    GROUP_SZE = 8

    persistent[(min(NUM_SM, triton.cdiv(M,Br) * triton.cdiv(N, Bc)), )](A, B, C,
                           A.stride(0),
                           B.stride(0),
                           C.stride(0),
                           NUM_SM, Br, Bc, Bk,
                           M, N, K, GROUP_SZE
                          ,num_stages = 4, num_warps = 8)
    
    return C
    
def test():
    M = N = K = 4096
    torch.manual_seed(20)
    dtype = torch.float16

    DEVICE = torch.device('cuda:0')
    properties = driver.active.utils.get_device_properties(DEVICE.index)
    NUM_SM = properties["multiprocessor_count"]

    A = torch.randn((M,K), dtype = dtype, device= DEVICE)
    B = torch.randn((K,N), dtype = dtype, device= DEVICE)
    C = torch.empty((M,N), dtype=torch.float16, device=DEVICE)
    
    ref_out = torch.matmul(A,B)
    tr_out = matmul_kernel(A,B,C, NUM_SM)

    print(f"correct? : {torch.allclose(tr_out, ref_out, atol=1e-1)}")
    print(f"dist : {torch.dist(tr_out, ref_out)}")


@triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[(2**i) for i in range(13)],
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
    C = torch.empty((M,N), dtype=torch.float16, device='cuda')
    
    DEVICE = torch.device('cuda:0')
    properties = driver.active.utils.get_device_properties(DEVICE.index)
    NUM_SM = properties["multiprocessor_count"]

    stream = getattr(torch, device.type).Stream()
    getattr(torch, device.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.matmul(A,B))
    
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: matmul_kernel(A,B,C, NUM_SM))
    gbps = lambda ms: ((2*N**3 - N**2)/1e9)/ (ms*1e-3)
    gbps = lambda ms: ms

    return gbps(ms)

if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)

    test()