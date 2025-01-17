import triton
import triton.language as tl
import torch
import torch.nn as nn
from layernorm2 import fast_ln2

@triton.jit
def layer_norm_fwd(Y, Y_row_stride,
                    X, X_row_stride,
                    W,
                    b,
                    r,
                    mu,
                    n_cols, eps,
                    BLOCK_SIZE : tl.constexpr):

    pid = tl.program_id(0)

    offset = tl.arange(0, BLOCK_SIZE)

    #TODO must generalize
    mask = offset < n_cols
    
    X += pid * X_row_stride
    Y += pid * Y_row_stride
    mu += pid
    r += pid

    x = tl.load(X + offset, mask=mask)
    w = tl.load(W + offset, mask=mask)
    b = tl.load(b + offset, mask=mask)

    mean = tl.sum(x)/BLOCK_SIZE
    num = x-mean
    var2 = tl.sum(num*num)/BLOCK_SIZE

    den = tl.math.sqrt(var2 + eps)

    preout = num/den
    out = preout * w + b

    tl.store(Y+offset, out, mask=mask)
    tl.store(mu, mean)
    tl.store(r, den)




class TritonLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, eps=1e-5):
        B,T,d = X.shape
        BLOCK_SIZE = d
        dtype = torch.float16
        Y = torch.empty((B,T,d), dtype=dtype,  device='cuda:0')
        x_row_stride = X.stride(-2)

        X = X.view(-1, d)
        r = torch.empty((d), dtype=dtype, device='cuda:0')
        mu = torch.empty((d), dtype=dtype, device='cuda:0')

        layer_norm_fwd[(B*T, )](Y, x_row_stride, X,
                              x_row_stride, W, b,
                              r, mu ,d, eps, BLOCK_SIZE)

        ctx.save_for_backward(X, W, mu, r)

        return Y


def fast_ln(X, ln):
    B,T,d = X.shape

    W = ln.weight
    b = ln.bias

    # fast_layer_norm = TritonLayerNorm()
    tr_out = TritonLayerNorm.apply(X, W, b)

    return tr_out.view(B,T,d)

    


def test(B = 16, T = 32, d = 64, dtype = torch.float16):
    device = 'cuda:0'

    X = torch.rand(size=(B,T,d), dtype=dtype, device=device)
    eps = 1e-5

    ln = nn.LayerNorm((d), eps = eps, dtype=dtype, device=device)
    torch.nn.init.uniform_(ln.weight)
    torch.nn.init.uniform_(ln.bias)
    ref_out = ln(X)

    tr_out = fast_ln(X, ln)

    print(f"dist: {torch.dist(ref_out, tr_out)}")
    assert torch.allclose(ref_out, tr_out, atol=1e-2, rtol=0.0)

    

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['d'],
        x_vals=[(2**i) for i in range(0, 15)],
        line_arg='provider',
        line_vals=['triton', 'torch', 'triton2'],
        line_names= ['Triton', 'Torch', 'Triton2'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],
        ylabel='ms',
        plot_name='perf',
        args = {}
    )
)

def benchmark(d, provider):
    torch.manual_seed(20)
    B = 64
    T = 64
    
    dtype = torch.float16
    device = torch.device('cuda')
    X = torch.rand(size=(B,T,d), dtype=dtype, device=device)
    eps = 1e-5
    stream = getattr(torch, device.type).Stream()
    getattr(torch, device.type).set_stream(stream)

    ln = nn.LayerNorm((d), eps = eps, dtype=dtype, device=device)
    torch.nn.init.uniform_(ln.weight)
    torch.nn.init.uniform_(ln.bias)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: ln(X))
    
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: fast_ln(X, ln))
        
    if provider == 'triton2':
        ms = triton.testing.do_bench(lambda: fast_ln2(X, ln))
         

    # gbps = lambda ms: 2 * X.numel() * X.element_size() * 1e-9 / (ms * 1e-3)
    gbps = lambda ms: ms

    return gbps(ms)




if __name__ == '__main__':
    benchmark.run(show_plots=True, print_data=True)