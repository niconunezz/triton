import torch
import triton
import triton.language as tl
from triton.runtime import driver
from grouping import kernel



@triton.autotune(
    configs=[
        triton.Config({"Br": 64, "Bc": 256, "Bk": 64, "GROUP_SZE": 8}, num_stages=3, num_warps=16),        
        triton.Config({"Br": 128, "Bc": 256, "Bk": 32, "GROUP_SZE": 4}, num_stages=3, num_warps=8),
        triton.Config({"Br": 64, "Bc": 32, "Bk": 64, "GROUP_SZE": 8}, num_stages=4, num_warps=8),
        triton.Config({"Br": 128, "Bc": 128, "Bk": 32, "GROUP_SZE": 4}, num_stages=5, num_warps=8),
        triton.Config({"Br": 128, "Bc": 128, "Bk": 64, "GROUP_SZE": 2}, num_stages=4, num_warps=8),

    ],
    key=["M", "N", "K"],
)

@triton.jit
def persistent(A, B, C,
               am_stride,
               bk_stride,
               cm_stride,
               NUM_SM: tl.constexpr,             
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               Br: tl.constexpr, Bc: tl.constexpr,Bk: tl.constexpr,
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
        a_ptrs = A + off_a
        b_ptrs = B + off_b
        # a = tl.load(a_ptrs, mask = k_off[None, :] < K - ki*Bk, other=0.0)
        # b = tl.load(b_ptrs, mask = k_off[:, None] < K - ki*Bk, other=0.0)

        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        accumulator = tl.dot(a, b, accumulator, allow_tf32=False)


        if ki == K_BLOCKS-1:
            if C.dtype.element_ty == tl.float8e4nv:
                c = accumulator.to(tl.float8e4nv)
            elif C.dtype.element_ty == tl.float8e5:
                c = accumulator.to(tl.float8e5)
            else:
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
    
    grid = lambda META: (min(NUM_SM, triton.cdiv(M,META['Br']) * triton.cdiv(N, META['Bc'])), )
    persistent[grid](A, B, C,
                     A.stride(0),
                     B.stride(0),
                     C.stride(0),
                     NUM_SM,
                     M, N, K)
   
    return C


    
def test():
    torch.manual_seed(20)
    dtype = torch.float16
    M = N = K = 2048 if dtype != torch.float16 else 4096

    DEVICE = torch.device('cuda:0')
    properties = driver.active.utils.get_device_properties(DEVICE.index)
    NUM_SM = properties["multiprocessor_count"]

    A = torch.randn((M,K), dtype = torch.float16, device= DEVICE)
    B = torch.randn((K,N), dtype = torch.float16, device= DEVICE)
    C = torch.empty((M,N), dtype=torch.float16, device=DEVICE)
    # ref_out = torch.matmul(A,B)
    
    A = A.to(dtype)
    B = B.to(dtype)
    C = C.to(dtype)
    tr_out = matmul_kernel(A,B,C, NUM_SM)

    # correct = torch.allclose(tr_out.to(torch.float16), ref_out, atol=1e-1)
    # dist = torch.dist(tr_out.to(torch.float16), ref_out)
    # mdist = torch.max(torch.abs((tr_out.to(torch.float16) - ref_out)))
    # print(f"correct? : {correct} | dist : {dist} | max dist : {mdist}")
    
    


@triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[(2**i) for i in range(13)],
            line_arg='provider',
            line_vals=['persistent', 'torch', 'grouping'],
            line_names= ['Persistent', 'Torch', 'Grouping'],
            styles=[('blue', '-'), ('green', '-'), ('red', '-')],
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
    A = torch.randn((M,K), dtype = torch.float16, device= device)
    B = torch.randn((K,N), dtype = torch.float16, device= device)
    C = torch.empty((M,N), dtype=torch.float16, device=device)
    
    DEVICE = torch.device('cuda:0')
    properties = driver.active.utils.get_device_properties(DEVICE.index)
    NUM_SM = properties["multiprocessor_count"]

    stream = getattr(torch, device.type).Stream()
    getattr(torch, device.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.matmul(A,B))

    A = A.to(dtype)
    B = B.to(dtype)
    C = C.to(dtype)

    if provider == 'persistent':
        ms = triton.testing.do_bench(lambda: matmul_kernel(A,B,C, NUM_SM))

    if provider == 'grouping':
        ms = triton.testing.do_bench(lambda: kernel(A,B,C))

    gbps = lambda ms: ((2*N**3 - N**2)/1e9)/ (ms*1e-3)

    return gbps(ms)

if __name__ == "__main__":
    
    benchmark.run(show_plots=True, print_data=True)

    # test()