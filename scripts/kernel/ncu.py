import argparse
import torch, gc
import cutlass

DEV = 'cuda:0'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=4096)
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--k', type=int, default=4096)
    parser.add_argument("--func", type=str, help="Function name to call")
    args = parser.parse_args()

    M = args.m
    N = args.n
    K = args.k

    # Retrieve function directly from cutlass
    func = getattr(cutlass, args.func, None)

    if "torch" in args.func :
        X = torch.randn((M, K), device=DEV, dtype=torch.half)
        W = torch.randn((N, K), device=DEV, dtype=torch.half)
        Y = torch.matmul(X, W.t())
        return
    
    is_custom = "custom" in args.func
    is_fp8 = "fp8" in args.func
    if callable(func):
        if is_fp8:
            X = torch.randn((M,K), device= DEV, dtype=torch.half)
            X8 = torch.tensor(X, device=DEV, dtype=torch.float8_e4m3fn)
            W = torch.randn((N,K), device = DEV, dtype=torch.half)
            W8 = torch.tensor(W, device=DEV, dtype=torch.float8_e4m3fn)
            Y = torch.zeros((N, M), device=DEV, dtype=torch.half)
            func(W8, X8, Y)
            Y = Y.view(M,N)
        elif not is_custom :
            # Profile tma warp specialized cooperative kernel
            X = torch.randn((M, K), device=DEV, dtype=torch.half)
            W = torch.randn((N, K), device=DEV, dtype=torch.half)
            Y = torch.zeros((N, M), device=DEV, dtype=torch.half)
            func(W, X, Y)
            Y = Y.view(M, N)
        else :
            # Profile tma warp specialized cooperative custom kernel
            X = torch.randn((M, K), device=DEV, dtype=torch.half)
            W = torch.randn((N, K), device=DEV, dtype=torch.half)
            W_upper = torch.empty(size=(N, K), device=DEV, dtype=torch.float8_e4m3fn)
            W_lower = torch.empty(size=(N, K), device=DEV, dtype=torch.float8_e4m3fn)
            W = torch.where(W.abs() > 1.75, 1.75, W)
            cutlass.divide_fp16(W, W_upper, W_lower)
            Y = torch.zeros((N, M), device=DEV, dtype=torch.half)
            func(W_upper, W_lower, X, Y)
            Y = Y.view(M, N)
    else:
        print(f"Error: Function '{args.func}' not found in 'cutlass'.")

if __name__ == "__main__":
    main()
