import triton
import triton.language as tl
import torch
import torch.nn as nn


@triton.jit
def layer_norm_fwd(Y, Y_row_stride,
                    X, X_row_stride,
                    W,
                    b_ptr,
                    r,
                    mu,
                    n_cols, eps,
                    BLOCK_SIZE : tl.constexpr):

    pid = tl.program_id(0)

    

    
    X += pid * X_row_stride
    Y += pid * Y_row_stride
    mu += pid
    r += pid

    sm = tl.zeros([BLOCK_SIZE,], dtype = tl.float16)
    offset = tl.arange(0, BLOCK_SIZE)
    for i in range(0, n_cols, BLOCK_SIZE):
        x = tl.load(X + offset, mask=offset < n_cols)
        sm += x
        offset += BLOCK_SIZE

    mean = tl.sum(sm)/ n_cols

    var = tl.zeros([BLOCK_SIZE,], dtype = tl.float16)
    offset = tl.arange(0, BLOCK_SIZE)

    for i in range(0, n_cols, BLOCK_SIZE):

        x = tl.load(X + offset, mask=offset < n_cols)
        x = tl.where(offset < n_cols, x-mean, 0.0).to(tl.float16)
        var += x*x

        offset += BLOCK_SIZE
    
    var = tl.sum(var) /n_cols
    rstd = 1 / tl.sqrt(var + eps)
    offset = tl.arange(0, BLOCK_SIZE)

    for i in range(0, n_cols, BLOCK_SIZE):

        x = tl.load(X + offset, mask=offset < n_cols)
        w = tl.load(W + offset, mask=offset < n_cols)
        b = tl.load(b_ptr + offset, mask=offset < n_cols)

        x_hat =(x-mean)*rstd
        y = x_hat * w + b

        tl.store(Y + offset, y, mask=offset < n_cols)

        offset += BLOCK_SIZE

   



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

        ctx.save_for_backward(r, mu)

        return Y


def fast_ln2(X, ln):
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

    tr_out = fast_ln2(X, ln)

    print(f"dist: {torch.dist(ref_out, tr_out)}")
    assert torch.allclose(ref_out, tr_out, atol=1e-2, rtol=0.0)

    

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['d'],
        x_vals=[(2**i) for i in range(0, 15)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names= ['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
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
    ref_out = ln(X)

    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: ln(X))
    
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: fast_ln2(X, ln))
         

    gbps = lambda ms: 2 * X.numel() * X.element_size() * 1e-9 / (ms * 1e-3)

    return gbps(ms)




if __name__ == '__main__':
    test()
    benchmark.run(show_plots=True, print_data=True)