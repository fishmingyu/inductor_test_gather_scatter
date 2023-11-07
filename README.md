## OpenMP test for SpMM

When compiling scatter_gather operator, PyTorch inductor will fall back to generate atomic-based code, which has large performance gap compared to CSR SpMM on CPU.
Here we propose a simple openmp code base to point out this phenomenon, paving the way for the future sparse compiler RFC.

Here is a simple demo for scatter_gather operator.

```python
def gather_scatter(x, edge_index, reduce="sum"):
    row, col = edge_index
    x_j = x[row]
    return scatter(x_j, col, dim_size=x.size(0), reduce=reduce)
```

### Code Running

Check if your pytorch version is larger than v2.0

**Codegen check**

```bash
cd benchmark
TORCH_COMPILE_DEBUG=1 python test_compile_basic.py
```

And then take a look at the `output_code.py` of your debug directory. (Under `torch_compile_debug/torchinductor`)

**Performance comparison:**

```bash
python spmm.py --dataset ['cora, etc.'] --feature [32, 64, ...]
```

### Result Demo

CPU info: (AVX, AVX2 supported)

```
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
Address sizes:                   39 bits physical, 48 bits virtual
CPU(s):                          16
On-line CPU(s) list:             0-15
Thread(s) per core:              2
Core(s) per socket:              8
Socket(s):                       1
NUMA node(s):                    1
Vendor ID:                       GenuineIntel
CPU family:                      6
Model:                           158
Model name:                      Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
```

![image info](./spmm.png)
