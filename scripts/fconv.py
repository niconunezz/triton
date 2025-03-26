import torch
import triton
import triton.language as tl
import torch.nn as nn
from grouping import kernel
import numpy as np


@triton.jit
def im2col_kernel(x_ptr, y_ptr, mask_ptr, k, stride,
                  H, W, n_slides, n_cols, mask,  M:tl.constexpr):
    
    # import pdb; pdb.set_trace()

    batch = tl.program_id(0)
    pid1 = tl.program_id(1)

    pid_n = pid1 % n_cols
    pid_m = pid1 // n_cols

    off = tl.arange(0, M)
    m_off = pid_m*stride + off
    n_off = pid_n*stride + off

    
    x_ptrs =x_ptr + batch*W*H + m_off[:, None]*W + n_off[None, :]

    x = tl.load(x_ptrs, eviction_policy='evict_last')#.reshape(M, M)

    y_off = pid_m*n_cols + pid_n
    y_ptrs = batch*(M*M)*n_slides + y_off*(M*M) + tl.arange(0, M)[:, None]*M + tl.arange(0, M)[None, :]
    
    tl.store(y_ptr + y_ptrs, x)

    # if mask:
    #     points = tl.arange(0, M*M)[None, :]
    #     mask_v = (((points+1) % M) != 0) & (points < M*k)
    #     tl.store(mask_ptr + y_ptrs, mask_v)


def im2col(x,y,mask_ptr, kernel, stride, m, n_slides, row_slides, col_slides, B, H, W) -> torch.Tensor:
    mask = True
    if m == kernel:
        mask_ptr = None
        mask = None

    im2col_kernel[(B,row_slides*col_slides,1)](x, y, mask_ptr, 
                                                          kernel, stride, 
                                                          H, W,
                                                          n_slides, col_slides,
                                                          mask,
                                                          m)

    return y, mask_ptr


def conv(x,y,C, mask, k:torch.Tensor, m, n_slides, row_slides, col_slides, B, H, W, out_shape, kernel_sze, mm):
    y, mask = im2col(x, y,mask, kernel=kernel_sze, stride=1, m=m, n_slides = n_slides, row_slides= row_slides, col_slides=col_slides,B=B, H=H, W=W)
    # print(y.shape)

    # if torch.is_tensor(mask):
    #     print(mask[0])
    #     y = y[mask].view(n_slides*B, kernel_sze**2)
    
    # if m != kernel_sze:
    #     y = y[mm].view(n_slides*B, kernel_sze**2)

    if m != kernel_sze:
        # y = y.view(n_slides*B, m, m)
        y = y[:, :kernel_sze, :kernel_sze]
    # print(mask)
    y = y.reshape(n_slides*B, kernel_sze**2)
    return kernel(y, k, C).view(B, 1, out_shape,out_shape)

def gen_data(kernel_sze, B, H, W, k):
    #! grows really fast
    m = triton.next_power_of_2(kernel_sze)

    row_slides, col_slides = (H - kernel_sze + 1),(W - kernel_sze +1)
    n_slides = row_slides * col_slides

    y = torch.empty((n_slides*B, m, m), device='cuda:0', dtype=torch.float16)
    mask_ptr = torch.empty((n_slides*B, m**2), device='cuda:0', dtype=torch.bool)

    out_shape = (W-kernel_sze)+1

    k = k.flatten().unsqueeze(0)

    device = torch.device('cuda:0')

    
    M,N = y.shape[0], k.shape[0]
    C = torch.empty((M,N), dtype=torch.float16, device=device)

    mask = torch.zeros((m,m), dtype=torch.bool, device='cuda')
    mask[:kernel_sze, :kernel_sze] = True
    mm = mask.flatten().reshape(1,-1).repeat(B*n_slides,1).to(device)

    return m, n_slides, row_slides, col_slides, y, mask_ptr, out_shape, k, C, mm



def test():
    B, C, H, W = 2, 1,32,32
    x = torch.randn((B,C, H, W), device = 'cuda:0', dtype=torch.float16)
    # x = torch.arange(0, 16).view(B,1,H,W).to('cuda:0').to(torch.float16)
    kernel_sze = 5
    convLayer = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=kernel_sze, stride = 1, padding=0,bias=False, dtype=torch.float16, device='cuda:0')
    k = convLayer.weight.data
    ref_out = convLayer(x)
    # print(f"ref out: {ref_out.shape}")

    m, n_slides, row_slides, col_slides, y, mask, out_shape, k, C, mm= gen_data(kernel_sze, B, H, W, k)
    tr_out = conv(x,y,C,mask, k, m, n_slides, row_slides, col_slides, B, H, W, out_shape, kernel_sze, mm)
    # print(f"tr out: {tr_out.shape}")
  
    print(f"correct: {torch.allclose(ref_out, tr_out, atol=1e-2)}")



@triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['kernel_sze'],
            x_vals=[(i) for i in range(1,12)],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names= ['Triton', 'Torch'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='ms',
            plot_name='perf',
            args = {}
    
        )
)

def benchmark(kernel_sze, provider):
    torch.manual_seed(20)
    device = torch.device('cuda:0')
    B = 1
    C = 1
    H = W = 32
    # kernel_sze = 4
    x = torch.randn((B,C, H, W), device = 'cuda:0', dtype=torch.float16)
    convLayer = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=kernel_sze, stride = 1, padding=0, bias=False, device='cuda:0', dtype=torch.float16)
    k = convLayer.weight.data
   

    m, n_slides, row_slides, col_slides, y, mask, out_shape, k, C, mm= gen_data(kernel_sze, B,H, W, k)


    stream = getattr(torch, device.type).Stream()
    getattr(torch, device.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: convLayer(x))

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: conv(x,y,C,mask, k, m, n_slides, row_slides, col_slides, B, H, W, out_shape, kernel_sze, mm))
    
    return ms

if __name__ == '__main__':
    test()
    benchmark.run(True, True)

