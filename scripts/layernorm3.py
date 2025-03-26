import triton
import triton.language as tl
import torch
import torch.nn as nn
from layernorm2 import fast_ln2
from layernorm import fast_ln
from triton.runtime import driver

@triton.jit
def layer_norm_fwd(Y, Y_row_stride,
                    X, X_row_stride,
                    W,
                    b,
                    r,
                    mu,
                    n_cols, eps,
                    M, NUM_SM,
                    BLOCK_SIZE : tl.constexpr):


    pid = tl.program_id(0)

    off2 = tl.arange(0, BLOCK_SIZE)
    rows_per_sm  = tl.cdiv(M, NUM_SM)
    
    for i in range(rows_per_sm):

        offset = pid * X_row_stride + tl.arange(0, BLOCK_SIZE) 
        mask = offset < M*X_row_stride


        x = tl.load(X + offset, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + off2).to(tl.float32)
        bias = tl.load(b + off2).to(tl.float32)

        mean = tl.sum(x)/BLOCK_SIZE
        num = x-mean
        var2 = tl.sum(num*num)/BLOCK_SIZE

        den = tl.math.sqrt(var2 + eps)

        preout = num/den
        out = preout * w + bias

        tl.store(Y+offset, out, mask)

        mu_ptr = mu + pid
        r_ptr = r + pid
        tl.store(mu_ptr, mean)
        tl.store(r_ptr, den)

        pid += NUM_SM




class TritonLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, eps=1e-5):
        B,T,d = X.shape
        BLOCK_SIZE = d
        Y = torch.empty((B*T,d), dtype=X.dtype,  device='cuda:0')
        x_row_stride = d

        DEVICE = torch.device('cuda:0')
        properties = driver.active.utils.get_device_properties(DEVICE.index)
        NUM_SM = properties["multiprocessor_count"]
        
        X = X.view(-1, d)
        r = torch.empty((d), device='cuda:0')
        mu = torch.empty((d), device='cuda:0')
        M = B*T
        layer_norm_fwd[(min(B*T, NUM_SM), )](Y, x_row_stride, X,
                              x_row_stride, W, b,
                              r, mu ,d, eps,
                              M, NUM_SM, 
                              BLOCK_SIZE)

        ctx.save_for_backward(X, W, mu, r)

        return Y


def fast_ln3(X, ln):
    B,T,d = X.shape

    W = ln.weight
    b = ln.bias

    tr_out = TritonLayerNorm.apply(X, W, b)

    return tr_out.view(B,T,d)

    


def test(B = 64, T = 128, d = 256):
    device = 'cuda:0'
    torch.manual_seed(20)
    X = torch.rand(size=(B,T,d), device=device)
    eps = 1e-5

    ln = nn.LayerNorm((d), eps = eps, device=device)
    torch.nn.init.uniform_(ln.weight)
    torch.nn.init.uniform_(ln.bias)
    ref_out = ln(X)

    # tr_out = fast_ln3(X, ln)
    # print(f"dist: {torch.dist(ref_out, tr_out)}")
    # assert torch.allclose(ref_out, tr_out, atol=1e-2, rtol=0.0)

    

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['d'],
        x_vals=[(2**i) for i in range(0, 12)],
        line_arg='provider',
        line_vals=['triton3', 'torch', 'triton2', 'triton'],
        line_names= ['Triton3', 'Torch', 'Triton2', 'Triton'],
        styles=[('blue', '-'), ('red', '-'), ('green', '-'), ('yellow', '-')],
        ylabel='ms',
        plot_name='perf',
        args = {}
    )
)

def benchmark(d, provider):
    torch.manual_seed(20)
    B = 64
    T = 64
    
    
    device = torch.device('cuda')
    X = torch.rand(size=(B,T,d), device=device)
    eps = 1e-5
    stream = getattr(torch, device.type).Stream()
    getattr(torch, device.type).set_stream(stream)

    ln = nn.LayerNorm((d), eps = eps, device=device)
    torch.nn.init.uniform_(ln.weight)
    torch.nn.init.uniform_(ln.bias)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: ln(X))
    
    if provider == 'triton3':
        ms = triton.testing.do_bench(lambda: fast_ln3(X, ln))
        
    if provider == 'triton2':
        ms = triton.testing.do_bench(lambda: fast_ln2(X, ln))

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: fast_ln(X, ln))
         


    return ms




if __name__ == '__main__':
    test()
    # benchmark.run(show_plots=True, print_data=True)