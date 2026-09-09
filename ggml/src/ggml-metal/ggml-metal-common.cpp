#include "ggml-metal-common.h"

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <vector>
#include <cstdlib>

bool ggml_metal_op_mul_mat_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne01 = op->src[0]->ne[1];
    const int64_t ne11 = op->src[1]->ne[1];

    return ne01 > 4 &&
           !ggml_is_transposed(op->src[0]) &&
           !ggml_is_transposed(op->src[1]) &&
           has_simdgroup_mm && ne00 >= 64 && ne11 > 8;
}

bool ggml_metal_op_mul_mat_id_use_mm(const struct ggml_tensor * op, bool has_simdgroup_mm) {
    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne21 = op->src[2]->ne[1];

    return has_simdgroup_mm && ne00 >= 64 && ne21 >= 32;
}

// represents a memory range (i.e. an interval from a starting address p0 to an ending address p1 in a given buffer pb)
// the type indicates whether it is a source range (i.e. ops read data from it) or a destination range (i.e. ops write data to it)
struct ggml_mem_range {
    uint64_t pb; // buffer id

    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

struct ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    int debug = 0;
};

ggml_mem_ranges_t ggml_mem_ranges_init(int debug) {
    auto * res = new ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    return res;
}

void ggml_mem_ranges_free(ggml_mem_ranges_t mrs) {
    delete mrs;
}

void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs) {
    mrs->ranges.clear();
}

static bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    mrs->ranges.push_back(mr);

    return true;
}

static ggml_mem_range ggml_mem_range_from_tensor(const ggml_tensor * tensor, ggml_mem_range_type pt) {
    // always use the base tensor
    tensor = tensor->view_src ? tensor->view_src : tensor;

    GGML_ASSERT(!tensor->view_src);

    ggml_mem_range mr;

    if (tensor->buffer) {
        // when the tensor is allocated, use the actual memory address range in the buffer
        //
        // take the actual allocated size with ggml_backend_buft_get_alloc_size()
        // this can be larger than the tensor size if the buffer type allocates extra memory
        // ref: https://github.com/ggml-org/llama.cpp/pull/15966
        mr = {
            /*.pb =*/ (uint64_t) tensor->buffer,
            /*.p0 =*/ (uint64_t) tensor->data,
            /*.p1 =*/ (uint64_t) tensor->data + ggml_backend_buft_get_alloc_size(tensor->buffer->buft, tensor),
            /*.pt =*/ pt,
        };
    } else {
        // otherwise, the pointer address is used as an unique id of the memory ranges
        //   that the tensor will be using when it is allocated
        mr = {
            /*.pb =*/ (uint64_t) tensor,
            /*.p0 =*/ 0,    //
            /*.p1 =*/ 1024, // [0, 1024) is a dummy range, not used
            /*.pt =*/ pt,
        };
    };

    return mr;
}

static ggml_mem_range ggml_mem_range_from_tensor_src(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_SRC);
}

static ggml_mem_range ggml_mem_range_from_tensor_dst(const ggml_tensor * tensor) {
    return ggml_mem_range_from_tensor(tensor, MEM_RANGE_TYPE_DST);
}

static bool ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

static bool ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range buf=%lld, [%lld, %lld)\n", __func__, mr.pb, mr.p0, mr.p1);
    }

    return ggml_mem_ranges_add(mrs, mr);
}

bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            ggml_mem_ranges_add_src(mrs, tensor->src[i]);
        }
    }

    return ggml_mem_ranges_add_dst(mrs, tensor);
}

static bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, ggml_mem_range mr) {
    for (size_t i = 0; i < mrs->ranges.size(); i++) {
        const auto & cmp = mrs->ranges[i];

        // two memory ranges cannot intersect if they are in different buffers
        if (mr.pb != cmp.pb) {
            continue;
        }

        // intersecting source ranges are allowed
        if (mr.pt == MEM_RANGE_TYPE_SRC && cmp.pt == MEM_RANGE_TYPE_SRC) {
            continue;
        }

        if (mr.p0 < cmp.p1 && mr.p1 >= cmp.p0) {
            if (mrs->debug > 2) {
                GGML_LOG_DEBUG("%s: the %s range buf=%lld, [%lld, %lld) overlaps with a previous %s range buf=%lld, [%lld, %lld)\n",
                        __func__,
                        mr.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        mr.pb, mr.p0, mr.p1,
                        cmp.pt == MEM_RANGE_TYPE_SRC ? "src" : "dst",
                        cmp.pb, cmp.p0, cmp.p1);
            }

            return false;
        }
    }

    return true;
}

static bool ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_src(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

static bool ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    GGML_ASSERT(tensor);

    ggml_mem_range mr = ggml_mem_range_from_tensor_dst(tensor);

    const bool res = ggml_mem_ranges_check(mrs, mr);

    return res;
}

bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (tensor->src[i]) {
            if (!ggml_mem_ranges_check_src(mrs, tensor->src[i])) {
                return false;
            }
        }
    }

    return ggml_mem_ranges_check_dst(mrs, tensor);
}

struct node_info {
    ggml_tensor * node;

    std::vector<ggml_tensor *> fused;
    bool track_all_writes = false;

    ggml_op op() const {
        return node->op;
    }

    const ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    bool is_empty() const {
        return ggml_op_is_empty(node->op);
    }

    void add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }
};

static bool ggml_metal_is_gdn_cache_cpy(const ggml_tensor * node) {
    if (node->op != GGML_OP_CPY || node->src[0] == nullptr) {
        return false;
    }

    const ggml_tensor * src = node->src[0];
    const ggml_tensor * gdn = src->view_src;
    if (src->op != GGML_OP_VIEW || gdn == nullptr || gdn->op != GGML_OP_GATED_DELTA_NET) {
        return false;
    }

    const ggml_tensor * v = gdn->src[2];
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, v->ne[0]*v->ne[1]*v->ne[2]*v->ne[3]);
    return src->view_offs == tail_off;
}

static std::vector<int> ggml_metal_graph_optimize_reorder(const std::vector<node_info> & nodes) {
    // helper to add node src and dst ranges
    const auto & h_add = [](ggml_mem_ranges_t mrs, const node_info & node) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node.node->src[i]) {
                if (!ggml_mem_ranges_add_src(mrs, node.node->src[i])) {
                    return false;
                }
            }
        }

        // keep track of the sources of the fused nodes as well
        for (const auto * fused : node.fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (fused->src[i]) {
                    if (!ggml_mem_ranges_add_src(mrs, fused->src[i])) {
                        return false;
                    }
                }
            }
        }

        if (node.track_all_writes) {
            if (!ggml_mem_ranges_add_dst(mrs, node.node)) {
                return false;
            }
            for (const auto * fused : node.fused) {
                if (!ggml_mem_ranges_add_dst(mrs, fused)) {
                    return false;
                }
            }
            return true;
        }
        return ggml_mem_ranges_add_dst(mrs, node.dst());
    };

    // helper to check if a node can run concurrently with the existing set of nodes
    const auto & h_check = [](ggml_mem_ranges_t mrs, const node_info & node) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node.node->src[i]) {
                if (!ggml_mem_ranges_check_src(mrs, node.node->src[i])) {
                    return false;
                }
            }
        }

        for (const auto * fused : node.fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (fused->src[i]) {
                    if (!ggml_mem_ranges_check_src(mrs, fused->src[i])) {
                        return false;
                    }
                }
            }
        }

        if (node.track_all_writes) {
            if (!ggml_mem_ranges_check_dst(mrs, node.node)) {
                return false;
            }
            for (const auto * fused : node.fused) {
                if (!ggml_mem_ranges_check_dst(mrs, fused)) {
                    return false;
                }
            }
            return true;
        }
        return ggml_mem_ranges_check_dst(mrs, node.dst());
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    const auto & h_safe = [](ggml_op op) {
        switch (op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
            case GGML_OP_ROPE:
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_L2_NORM:
            case GGML_OP_SUM_ROWS:
            case GGML_OP_SSM_CONV:
            case GGML_OP_SSM_SCAN:
            case GGML_OP_CLAMP:
            case GGML_OP_TRI:
            case GGML_OP_DIAG:
            case GGML_OP_MUL:
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
            case GGML_OP_GLU:
            case GGML_OP_SCALE:
            case GGML_OP_UNARY:
            case GGML_OP_GET_ROWS:
            case GGML_OP_SET_ROWS:
            case GGML_OP_SET:
            case GGML_OP_CPY:
            case GGML_OP_CONT:
            case GGML_OP_REPEAT:
                return true;
            default:
                return ggml_op_is_empty(op);
        }
    };

    const int n = nodes.size();

    std::vector<int> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    // the memory ranges for the set of currently concurrent nodes
    ggml_mem_ranges_t mrs0 = ggml_mem_ranges_init(0);

    // the memory ranges for the set of nodes that haven't been processed yet, when looking forward for a node to reorder
    ggml_mem_ranges_t mrs1 = ggml_mem_ranges_init(0);

    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) {
            continue;
        }

        const auto & node0 = nodes[i0];

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        //
        // note: we can always add empty nodes to the concurrent set as they don't read nor write anything
        if (!node0.is_empty() && !h_check(mrs0, node0)) {
            if (ggml_metal_is_gdn_cache_cpy(node0.node)) {
                ggml_mem_ranges_reset(mrs0);
                h_add(mrs0, node0);
                res.push_back(i0);
                continue;
            }

            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            // if a node is not concurrent with this set, we cannot reorder it
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            // that many nodes forward to search for a concurrent node
            constexpr int N_FORWARD = 64;

            for (int i1 = i0 + 1; i1 < i0 + N_FORWARD && i1 < n; i1++) {
                if (used[i1]) {
                    continue;
                }

                const auto & node1 = nodes[i1];

                // disallow reordering of certain ops
                if (!h_safe(node1.op())) {
                    break;
                }

                const bool is_empty = node1.is_empty();

                // to reorder a node and add it to the concurrent set, it has to be:
                //   + empty or concurrent with all nodes in the existing concurrent set (mrs0)
                //   + concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                if ((is_empty || h_check(mrs0, node1)) && h_check(mrs1, node1)) {
                    // add the node to the existing concurrent set (i.e. reorder it for early execution)
                    h_add(mrs0, node1);
                    res.push_back(i1);

                    // mark as used, so we skip re-processing it later
                    used[i1] = true;
                } else {
                    // expand the set of nodes that haven't been processed yet
                    h_add(mrs1, node1);
                }
            }

            // finalize the concurrent set and begin a new one
            ggml_mem_ranges_reset(mrs0);
        }

        // expand the concurrent set with the current node
        {
            h_add(mrs0, node0);
            res.push_back(i0);
        }
    }

    ggml_mem_ranges_free(mrs0);
    ggml_mem_ranges_free(mrs1);

    return res;
}

// Keep implemented fusions atomic during the concurrency reorder. Every write
// is tracked because an encoder boundary or a backend shape guard can still
// select the unfused path. This also covers stateful fusions with two outputs.
static int ggml_metal_preserve_fusion_span(const ggml_cgraph * gf, int begin) {
    const ggml_tensor * first = gf->nodes[begin];
    if (begin + 1 >= gf->n_nodes) {
        return 1;
    }
    static const bool snapshots=std::getenv("GGML_METAL_CONV_SNAPSHOT_FUSE")!=nullptr;
    if (snapshots && first->op==GGML_OP_CONCAT && first->ne[0]<=16) {
        int last=begin, copies=0;
        for (int j=begin+1;j<gf->n_nodes && j<begin+32 && copies<8;++j) {
            const ggml_tensor * node=gf->nodes[j];
            if (ggml_is_empty(node) || node->op==GGML_OP_VIEW || node->op==GGML_OP_RESHAPE ||
                node->op==GGML_OP_PERMUTE || node->op==GGML_OP_TRANSPOSE) continue;
            if (node->op!=GGML_OP_CPY || !node->src[0] || node->src[0]->view_src!=first) break;
            last=j;
            ++copies;
        }
        // Every member stays observable; the scheduler tracks all writes.
        if (copies>=2) return last-begin+1;
    }
    const ggml_tensor * second = gf->nodes[begin + 1];
    static const int gateup_rows = std::getenv("GGML_METAL_EXPERT_GATEUP") ? std::atoi(std::getenv("GGML_METAL_EXPERT_GATEUP")) : 0;
    if ((gateup_rows==1 || gateup_rows==2 || gateup_rows==4) && first->op==GGML_OP_MUL_MAT_ID &&
        first->src[0]->type==GGML_TYPE_Q4_K && first->src[2]->ne[1]<=8 && begin+2<gf->n_nodes) {
        const ggml_tensor * last=gf->nodes[begin+2];
        const ggml_op ops[]={GGML_OP_MUL_MAT_ID,GGML_OP_MUL_MAT_ID,GGML_OP_GLU};
        const int output=begin+2;
        if (second->op==GGML_OP_MUL_MAT_ID && last->op==GGML_OP_GLU &&
            ggml_get_glu_op(last)==GGML_GLU_OP_SWIGLU &&
            ggml_can_fuse_subgraph(gf,begin,3,ops,&output,1)) {
            return 3;
        }
    }
    static const bool rms_gate = std::getenv("GGML_METAL_RMS_GATE_FUSE") != nullptr;
    if (rms_gate && first->op == GGML_OP_RMS_NORM) {
        const ggml_op expected[] = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_UNARY, GGML_OP_MUL };
        ggml_op ops[16];
        int matched = 0;
        int end = begin;
        for (; end < gf->n_nodes && end < begin + 16 && matched < 4; ++end) {
            const auto * node = gf->nodes[end];
            ops[end - begin] = node->op;
            if (ggml_op_is_empty(node->op)) {
                continue;
            }
            if (node->op != expected[matched] ||
                    (matched == 2 && ggml_get_unary_op(node) != GGML_UNARY_OP_SIGMOID)) {
                break;
            }
            ++matched;
        }
        const int count = end - begin;
        const int output = begin + count - 1;
        if (matched == 4 && ggml_can_fuse_subgraph(gf, begin, count, ops, &output, 1)) {
            return count;
        }
    }
    if ((first->op == GGML_OP_RMS_NORM || first->op == GGML_OP_NORM) &&
            second->op == GGML_OP_SCALE &&
            ggml_can_fuse(gf, begin, { first->op, GGML_OP_SCALE })) {
        return 2;
    }
    if (first->op == GGML_OP_SSM_CONV && second->op == GGML_OP_UNARY &&
            std::getenv("GGML_METAL_CONV_SILU_FUSE") != nullptr &&
            ggml_get_unary_op(second) == GGML_UNARY_OP_SILU &&
            ggml_can_fuse(gf, begin, { GGML_OP_SSM_CONV, GGML_OP_UNARY })) {
        return 2;
    }
    if (first->op == GGML_OP_SCALE && second->op == GGML_OP_UNARY &&
            ggml_get_unary_op(second) == GGML_UNARY_OP_SILU &&
            ggml_can_fuse(gf, begin, { GGML_OP_SCALE, GGML_OP_UNARY })) {
        return 2;
    }

    int next = begin + 1;
    while (next < gf->n_nodes && next < begin + 16 && ggml_op_is_empty(gf->nodes[next]->op)) {
        ++next;
    }
    if (next >= gf->n_nodes || next >= begin + 16) {
        return 1;
    }
    if (first->op == GGML_OP_GATED_DELTA_NET && ggml_metal_is_gdn_cache_cpy(gf->nodes[next]) &&
            gf->nodes[next]->src[0]->view_src == first) {
        return next - begin + 1;
    }

    int sigmoid = begin;
    if (first->op == GGML_OP_MUL_MAT) {
        sigmoid = next;
    }
    if (gf->nodes[sigmoid]->op != GGML_OP_UNARY ||
            ggml_get_unary_op(gf->nodes[sigmoid]) != GGML_UNARY_OP_SIGMOID) {
        return 1;
    }
    next = sigmoid + 1;
    while (next < gf->n_nodes && next < begin + 16 && ggml_op_is_empty(gf->nodes[next]->op)) {
        ++next;
    }
    if (next >= gf->n_nodes || next >= begin + 16 ||
            gf->nodes[next]->op != GGML_OP_QWEN4EXP_HC_REDUCE ||
            gf->nodes[next]->src[1] != gf->nodes[sigmoid]) {
        return 1;
    }
    const int count = next - begin + 1;
    ggml_op ops[16];
    for (int i = 0; i < count; ++i) {
        ops[i] = gf->nodes[begin + i]->op;
    }
    const int output = begin + count - 1;
    return ggml_can_fuse_subgraph(gf, begin, count, ops, &output, 1) ? count : 1;
}

void ggml_graph_optimize(ggml_cgraph * gf) {
    constexpr int MAX_FUSE = 16;
    static const bool preserve_fusions = std::getenv("GGML_METAL_FUSION_AWARE_SCHEDULE") != nullptr;

    const int n = gf->n_nodes;

    enum ggml_op ops[MAX_FUSE];

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we first pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    for (int i = 0; i < n; i++) {
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.fused =*/ {},
        };

        if (preserve_fusions) {
            const int count = ggml_metal_preserve_fusion_span(gf, i);
            if (count > 1) {
                node.track_all_writes = true;
                for (int k = 1; k < count; ++k) {
                    node.add_fused(gf->nodes[++i]);
                }
                nodes.push_back(std::move(node));
                continue;
            }
        }

        // fuse only ops that start with these operations
        // can be expanded when needed
        if (node.op() == GGML_OP_ADD ||
            node.op() == GGML_OP_NORM ||
            node.op() == GGML_OP_RMS_NORM) {
            ops[0] = node.op();

            int f = i + 1;
            while (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                if (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_NORM &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    break;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            for (; f > 1; f--) {
                if (ggml_can_fuse(gf, i, ops, f)) {
                    break;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            for (int k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                node.add_fused(gf->nodes[i]);
            }
        }

        nodes.push_back(std::move(node));
    }

#if 1
    // reorder to improve concurrency
    const auto order = ggml_metal_graph_optimize_reorder(nodes);
#else
    std::vector<int> order(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        order[i] = i;
    }
#endif

    // unfuse
    {
        int j = 0;
        for (const auto i : order) {
            const auto & node = nodes[i];

            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
