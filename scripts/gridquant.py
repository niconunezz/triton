import triton
import triton.language as tl
import torch

@triton.jit
def block_quantization(A, 
                       A_row_stride,
                       A_float_8,
                       scale_matrix_ptr,
                       M:tl.constexpr,
                       N:tl.constexpr,
                       NUM_MSBLOCKS:tl.constexpr,
                       NUM_SBLOCKS:tl.constexpr,
                       GROUP_SZE:tl.constexpr,
                       BLOCK_SZE:tl.constexpr):
    
    pid = tl.program_id(0)

    n_blocks = tl.cdiv(N, BLOCK_SZE)
    pid_m = pid//n_blocks
    pid_n = pid%n_blocks

    block_off_m = pid_m*BLOCK_SZE*A_row_stride
    block_off_n = pid_n*BLOCK_SZE
    block_off = block_off_m + block_off_n
    A += block_off
    A_float_8 += block_off
    max_vals = tl.zeros([GROUP_SZE, GROUP_SZE], dtype = tl.bfloat16)

    for i in range(0, NUM_SBLOCKS):
        
        # position in subgroup
        curr_m = i // NUM_MSBLOCKS
        curr_n = i % NUM_MSBLOCKS

        off = tl.arange(0, GROUP_SZE)
        off_m = (off[:, None] + curr_m*GROUP_SZE)*A_row_stride
        off_n = off[None, :] + curr_n*GROUP_SZE
        offset = off_m + off_n
        mask =  (((off_m + block_off_m) < M*A_row_stride) & ((off_n+block_off_n) < N))
        block = tl.load(A + offset, mask=mask, other=0.0, eviction_policy='evict_last')
        max_vals = tl.maximum(max_vals,tl.abs(block)).to(tl.bfloat16)
    
    sf = tl.max(max_vals).to(tl.bfloat16)

    for i in range(0, NUM_SBLOCKS):
        
        curr_m = i // NUM_MSBLOCKS
        curr_n = i % NUM_MSBLOCKS

        off = tl.arange(0, GROUP_SZE)
        off_m = (off[:, None] + curr_m*GROUP_SZE)*A_row_stride
        off_n = off[None, :] + curr_n*GROUP_SZE
        offset = off_m + off_n
        mask =  (((off_m + block_off_m) < M*A_row_stride) & ((off_n+block_off_n) < N))
        block = tl.load(A + offset, mask=mask, other=0.0, eviction_policy='evict_last')

        scaled_matrix = block * 1/sf
        fp8 = tl.cast(scaled_matrix, tl.float8e5, fp_downcast_rounding='rtne')
        tl.store(A_float_8 + offset, fp8 , mask=mask)
    
    tl.store(scale_matrix_ptr + pid_m*n_blocks + pid_n, sf)


def quantization(A, B, BLOCK_SZE, GROUP_SZE):

    device = 'cuda'
    M, N = A.shape
    
    M_BLOCKS = triton.cdiv(M,BLOCK_SZE) # blocks on each axis
    NUM_SMBLOCKS = triton.cdiv(BLOCK_SZE,GROUP_SZE) # subblocks on each axis
    NUM_SBLOCKS = NUM_SMBLOCKS**2 # total subblocks
    
    A_float8 = torch.empty((M,N), dtype=torch.float8_e5m2, device = device)
    scale_a = torch.empty((M_BLOCKS,M_BLOCKS), dtype=torch.bfloat16, device=device)
    B_float8 = torch.empty((M,N), dtype=torch.float8_e5m2, device = device)
    scale_b = torch.empty((M_BLOCKS,M_BLOCKS), dtype=torch.bfloat16, device=device)

    block_quantization[((triton.cdiv(M, BLOCK_SZE) * triton.cdiv(N, BLOCK_SZE)), )](A, A.stride(0),
                                                                                    A_float8, scale_a, M, N, NUM_SMBLOCKS,
                                                                                    NUM_SBLOCKS, GROUP_SZE, BLOCK_SZE)
    
    block_quantization[((triton.cdiv(M, BLOCK_SZE) * triton.cdiv(N, BLOCK_SZE)), )](B, B.stride(0),
                                                                                            B_float8, scale_b, M, N, NUM_SMBLOCKS,
                                                                                            NUM_SBLOCKS, GROUP_SZE, BLOCK_SZE)
    

    return A_float8, scale_a, B_float8, scale_b


# torch implementation to check
def normal(A, B, BLOCK_SZE, GROUP_SZE):
    M,N = A.shape
  
    BLOCKS_M = triton.cdiv(M,BLOCK_SZE)

    scale_a = torch.empty((BLOCKS_M,BLOCKS_M), dtype=torch.bfloat16)
    scale_b = torch.empty((BLOCKS_M,BLOCKS_M), dtype=torch.bfloat16)

    A_float8 = torch.empty((M,N), dtype=torch.float8_e5m2)
    B_float8 = torch.empty((M,N), dtype=torch.float8_e5m2)

    for i in range(BLOCKS_M):
        for j in range(BLOCKS_M):
            
            max_a = torch.max(torch.abs(A[i*BLOCK_SZE:(i+1)*BLOCK_SZE, j*BLOCK_SZE:(j+1)*BLOCK_SZE]))
            max_b = torch.max(torch.abs(B[i*BLOCK_SZE:(i+1)*BLOCK_SZE, j*BLOCK_SZE:(j+1)*BLOCK_SZE]))
            scale_a[i,j] = max_a
            scale_b[i,j] = max_b

            A_float8[i*BLOCK_SZE:(i+1)*BLOCK_SZE, j*BLOCK_SZE:(j+1)*BLOCK_SZE] =\
                 (A[i*BLOCK_SZE:(i+1)*BLOCK_SZE, j*BLOCK_SZE:(j+1)*BLOCK_SZE] * 1/max_a)
            B_float8[i*BLOCK_SZE:(i+1)*BLOCK_SZE, j*BLOCK_SZE:(j+1)*BLOCK_SZE] =\
                (B[i*BLOCK_SZE:(i+1)*BLOCK_SZE, j*BLOCK_SZE:(j+1)*BLOCK_SZE] * 1/max_b)

    A_float8, scale_a, B_float8, scale_b = A_float8.to('cuda'), scale_a.to('cuda'), B_float8.to('cuda'), scale_b.to('cuda')
    return A_float8, scale_a, B_float8, scale_b


def test():
    torch.manual_seed(20)
    M = N = 256
    BLOCK_SZE = 64
    GROUP_SZE = 16
    
    A = torch.randn((M,N), dtype=torch.bfloat16, device='cuda')
    B = torch.randn((M,N), dtype=torch.bfloat16, device='cuda')

    ref_A, ref_scale_a, ref_B, ref_scale_b = normal(A, B, BLOCK_SZE, GROUP_SZE)
    A_float8, scale_a, B_float8, scale_b = quantization(A, B, BLOCK_SZE,GROUP_SZE)

    assert torch.allclose(scale_a, ref_scale_a), 'scale matrix A is wrong'
    assert torch.allclose(scale_b, ref_scale_b), 'scale matrix B is wrong'