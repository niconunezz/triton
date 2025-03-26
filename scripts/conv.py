import triton 
import triton.language as tl
import torch
import torch.nn as nn

@triton.jit
def conv_kern(x_ptr, k_ptr, z_ptr, 
         N0, H, W, 
         KH: tl.constexpr, 
         KW: tl.constexpr, 
         B0: tl.constexpr):
    # import pdb; pdb.set_trace()
    #todo change
    pid_0 = tl.program_id(0)
    b_off = pid_0*H*W

    k_ptrs = k_ptr + tl.arange(0, KH)[:, None]*KW + tl.arange(0, KW)[None, :]
    k = tl.load(k_ptrs)

    n_slides = (H - KH + 1) * (W - KW +1)
    z_b_off = pid_0 * n_slides
    for i in range(n_slides):
        row = i // (W - KW +1)
        col = i % (W - KW +1)

        row_off = row + tl.arange(0, KH)
        col_off = col + tl.arange(0, KW)
        x_off = row_off[:, None] * W + col_off[None, :]

        mask = (row_off < W)[:, None] & (col_off < H)[None, :]
        x_ptrs = x_ptr + b_off + x_off
        x = tl.load(x_ptrs, mask=mask)
        out = tl.sum(x*k)
        z_ptrs = z_ptr + z_b_off + i
        tl.store(z_ptrs, out)
        

def conv(x, k, z):
    KH, KW =  k.squeeze().shape
    B, C, H, W = x.shape
    conv_kern[(B, )](x, k, z, B, H, W, KH, KW, 1)

    return z


def test():
    B, C, H, W = 4,1,512, 512
    x = torch.randn((B,C, H, W), device = 'cuda:0', dtype=torch.float32)
    KW = KH = 2

    z = torch.empty((B, 1, (H - KH + 1), (W - KW +1)), device='cuda:0', dtype=torch.float32)
    convLayer = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=KW, stride = 1, padding=0,bias=False, device='cuda:0')
    k = convLayer.weight.data
    ref_out = convLayer(x)
    tr_out = conv(x, k, z)
  
    print(f"correct: {torch.allclose(ref_out, tr_out)}")
   


@triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['B'],
            x_vals=[(2**i) for i in range(7)],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names= ['Triton', 'Torch'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='ms',
            plot_name='perf',
            args = {}
    
        )
)

def benchmark(B, provider):
    torch.manual_seed(20)
    device = torch.device('cuda:0')
    C, H, W = 1,512, 512
    x = torch.randn((B,C, H, W), device = 'cuda:0', dtype=torch.float32)
    KW = KH = 2

    z = torch.empty((B, 1, (H - KH + 1), (W - KW +1)), device='cuda:0', dtype=torch.float32)
    convLayer = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=KW, stride = 1, padding=0, bias=False, device='cuda:0')
    k = convLayer.weight.data

    stream = getattr(torch, device.type).Stream()
    getattr(torch, device.type).set_stream(stream)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: convLayer(x))


    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: conv(x, k, z))
    

    return ms

if __name__ == '__main__':
    benchmark.run(True, True)