import triton
import triton.language as tl
import numpy as np


@triton.jit
def spmm_csr(A_ptr, A_ind, B, C, feature_size: tl.constexpr):
    # Global index corresponds to the node id
    node_id = tl.program_id(0)

    # Use tl.arange to get the feature id for each thread within the block
    feature_id = tl.arange(0, feature_size)

    # Using a local temporary variable to accumulate results
    acc = tl.load(C + node_id * feature_size + feature_id)

    # CSR loop for the specific node
    start = tl.load(A_ptr + node_id)
    end = tl.load(A_ptr + node_id + 1)
    for j in range(start, end):
        col = tl.load(A_ind + j)
        acc += tl.load(B + col * feature_size + feature_id)

    # Store the result back to C using tl.store
    tl.store(C + node_id * feature_size + feature_id, acc)


@triton.jit
def spmm_atomic(edge_index, B, C, num_edges, feature_size: tl.constexpr, XBLOCK: tl.constexpr):
    group_id = tl.program_id(0)
    xoffset = group_id * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    x1 = xindex // feature_size
    x2 = xindex % feature_size
    mask = x1 < num_edges
    in_node = tl.load(edge_index + x1, mask)
    out_node = tl.load(edge_index + x1 + num_edges, mask)
    in_val = tl.load(B + in_node * feature_size + x2, mask)
    tl.atomic_add(C + out_node * feature_size + x2, in_val, mask)


def spmm_atomic_wrapper(edge_index, B, C):
    feature_size = B.shape[1]
    num_edges = edge_index.shape[1]
    XBLOCK = 128
    spmm_atomic[(feature_size * num_edges // XBLOCK, )](edge_index, B, C, num_edges,
                      feature_size, XBLOCK=XBLOCK)


def spmm_csr_wrapper(rowptr, col, B, C):
    feature_size = B.shape[1]
    num_nodes = rowptr.shape[0] - 1
    spmm_csr[(num_nodes,)](rowptr, col, B, C, feature_size)
