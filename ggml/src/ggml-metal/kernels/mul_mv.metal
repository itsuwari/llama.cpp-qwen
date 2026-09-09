#include "common.h"
#include "dequantize.h"

// Copy every convolution rollback window in one dispatch, including repeated
// prefix windows when the rollback capacity exceeds this decode batch.
kernel void kernel_mul_mv_conv_snapshots(
        constant ggml_metal_kargs_cpy & a,
        device const char * input,
        device char * output,
        constant uint64_t * p,
        uint3 gid [[thread_position_in_grid]]) {
    const uint64_t i = gid.x;
    const uint64_t count = (uint64_t)a.ne00*a.ne01*a.ne02;
    if (i >= count || gid.y >= p[0]) return;
    const uint64_t r = i%a.ne00;
    const uint64_t ch = (i/a.ne00)%a.ne01;
    const uint64_t seq = i/((uint64_t)a.ne00*a.ne01);
    const uint64_t start = p[1] > gid.y ? p[1]-gid.y : 0;
    const uint64_t src = (start+r)*a.nb00 + ch*a.nb01 + seq*a.nb02;
    const uint64_t dst = (uint64_t)gid.y*p[2] + seq*a.nb1 + (ch*a.ne00+r)*a.nb0;
    *(device uint32_t *)(output+dst) = *(device const uint32_t *)(input+src);
}

// Q1_0 dot product: dot = d * (2 * Σ(yl[i] where bit=1) - sumy)
inline float block_q_n_dot_y(device const block_q1_0 * qb_curr, float sumy, thread float * yl, int il) {
    device const uint8_t * qs = qb_curr->qs + il / 8;
    const uint8_t b0 = qs[0];
    const uint8_t b1 = qs[1];

    float acc = 0.0f;

    acc += select(0.0f, yl[ 0], bool(b0 & 0x01));
    acc += select(0.0f, yl[ 1], bool(b0 & 0x02));
    acc += select(0.0f, yl[ 2], bool(b0 & 0x04));
    acc += select(0.0f, yl[ 3], bool(b0 & 0x08));
    acc += select(0.0f, yl[ 4], bool(b0 & 0x10));
    acc += select(0.0f, yl[ 5], bool(b0 & 0x20));
    acc += select(0.0f, yl[ 6], bool(b0 & 0x40));
    acc += select(0.0f, yl[ 7], bool(b0 & 0x80));

    acc += select(0.0f, yl[ 8], bool(b1 & 0x01));
    acc += select(0.0f, yl[ 9], bool(b1 & 0x02));
    acc += select(0.0f, yl[10], bool(b1 & 0x04));
    acc += select(0.0f, yl[11], bool(b1 & 0x08));
    acc += select(0.0f, yl[12], bool(b1 & 0x10));
    acc += select(0.0f, yl[13], bool(b1 & 0x20));
    acc += select(0.0f, yl[14], bool(b1 & 0x40));
    acc += select(0.0f, yl[15], bool(b1 & 0x80));

    return qb_curr->d * (2.0f * acc - sumy);
}

// Q2_0 dot: d * (sum_lo(y) + 2*sum_hi(y) - sumy) via per-bit conditional adds
inline float block_q_n_dot_y(device const block_q2_0 * qb_curr, float sumy, thread float * yl, int il) {
    device const uint8_t * qs = qb_curr->qs + (il / 4);
    const uint8_t b0 = qs[0];
    const uint8_t b1 = qs[1];
    const uint8_t b2 = qs[2];
    const uint8_t b3 = qs[3];

    // Accumulate where low bit is set (bits 0,2,4,6 of each byte)
    float acc_lo = 0.0f;
    acc_lo += select(0.0f, yl[ 0], bool(b0 & 0x01));
    acc_lo += select(0.0f, yl[ 1], bool(b0 & 0x04));
    acc_lo += select(0.0f, yl[ 2], bool(b0 & 0x10));
    acc_lo += select(0.0f, yl[ 3], bool(b0 & 0x40));
    acc_lo += select(0.0f, yl[ 4], bool(b1 & 0x01));
    acc_lo += select(0.0f, yl[ 5], bool(b1 & 0x04));
    acc_lo += select(0.0f, yl[ 6], bool(b1 & 0x10));
    acc_lo += select(0.0f, yl[ 7], bool(b1 & 0x40));
    acc_lo += select(0.0f, yl[ 8], bool(b2 & 0x01));
    acc_lo += select(0.0f, yl[ 9], bool(b2 & 0x04));
    acc_lo += select(0.0f, yl[10], bool(b2 & 0x10));
    acc_lo += select(0.0f, yl[11], bool(b2 & 0x40));
    acc_lo += select(0.0f, yl[12], bool(b3 & 0x01));
    acc_lo += select(0.0f, yl[13], bool(b3 & 0x04));
    acc_lo += select(0.0f, yl[14], bool(b3 & 0x10));
    acc_lo += select(0.0f, yl[15], bool(b3 & 0x40));

    // Accumulate where high bit is set (bits 1,3,5,7 of each byte)
    float acc_hi = 0.0f;
    acc_hi += select(0.0f, yl[ 0], bool(b0 & 0x02));
    acc_hi += select(0.0f, yl[ 1], bool(b0 & 0x08));
    acc_hi += select(0.0f, yl[ 2], bool(b0 & 0x20));
    acc_hi += select(0.0f, yl[ 3], bool(b0 & 0x80));
    acc_hi += select(0.0f, yl[ 4], bool(b1 & 0x02));
    acc_hi += select(0.0f, yl[ 5], bool(b1 & 0x08));
    acc_hi += select(0.0f, yl[ 6], bool(b1 & 0x20));
    acc_hi += select(0.0f, yl[ 7], bool(b1 & 0x80));
    acc_hi += select(0.0f, yl[ 8], bool(b2 & 0x02));
    acc_hi += select(0.0f, yl[ 9], bool(b2 & 0x08));
    acc_hi += select(0.0f, yl[10], bool(b2 & 0x20));
    acc_hi += select(0.0f, yl[11], bool(b2 & 0x80));
    acc_hi += select(0.0f, yl[12], bool(b3 & 0x02));
    acc_hi += select(0.0f, yl[13], bool(b3 & 0x08));
    acc_hi += select(0.0f, yl[14], bool(b3 & 0x20));
    acc_hi += select(0.0f, yl[15], bool(b3 & 0x80));

    return qb_curr->d * (acc_lo + 2.0f * acc_hi - sumy);
}

// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);

    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 2 + il/2);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// function for calculate inner product between half a q5_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 3 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (sumy * -16.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// function for calculate inner product between half a q5_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_1/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 4 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    constexpr short NW = N_SIMDWIDTH;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];
constant short FC_mul_mv_ne12  [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2    [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3    [[function_constant(FC_MUL_MV + 4)]];
constant bool  FC_mul_mv_split [[function_constant(FC_MUL_MV + 5)]];
constant bool  FC_mul_mv_id_tile_order [[function_constant(FC_MUL_MV + 10)]];
constant short FC_mul_mv_cache_tile [[function_constant(FC_MUL_MV + 6)]];
constant bool FC_mul_mv_expert_sort [[function_constant(FC_MUL_MV + 7)]];
constant short FC_mul_mv_expert_sort_tile [[function_constant(FC_MUL_MV + 8)]];
constant short FC_mul_mv_vector_load [[function_constant(FC_MUL_MV + 9)]];

template<typename block_q_type, short NR0, typename args_t>
void mul_vec_q_n_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 16;

    const int nb = args.ne00/QK4_0;

    const int r0 = (tgpig.x*NSG + sgitg)*NR0;
  //const int r0 =  tgpig.x*NR0;
    const int r1 =  tgpig.y;
    const int im =  tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const block_q_type * x = (device const block_q_type *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q_type * ax[NR0];
    FOR_UNROLL (int row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax[row] = (device const block_q_type *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = {0.f};

    const short ix = (tiisg/(NW/NQ));
    const short il = (tiisg%(NW/NQ))*8;

    //const int ib0 = sgitg*NQ + ix;
    const int ib0 = ix;

    float yl[16]; // src1 vector cache

    //device const float * yb = y + ix*QK4_0 + il;
    device const float * yb = y + ib0*QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    //for (int ib = ib0; ib < nb; ib += NSG*NQ) {
    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = { 0.f, 0.f };

        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;

            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * 16;
        //yb += NSG*NQ*QK4_0;
    }

    device float * dst_f32 = (device float *) dst + im*args.ne0*args.ne1 + r1*args.ne0;

    //helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && r0 + row < args.ne01) {
            dst_f32[r0 + row] = tot;
        }
    }
}

template<int nr0, typename args_t>
void kernel_mul_mv_q1_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK1_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q1_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_q1_0 *) ((device char *) src0 + offset0);
    }

    float yl[16];
    float sumf[nr0] = {0.f};

    const short ix = (tiisg/8);
    const short il = (tiisg%8)*16;

    device const float * yb = y + ix*QK1_0 + il;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/8) {
        float sumy = 0.f;

        FOR_UNROLL (short i = 0; i < 16; i++) {
            yl[i] = yb[i];
            sumy += yb[i];
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy, yl, il);
        }

        yb += QK1_0 * (N_SIMDWIDTH/8);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q1_0_f32")]]
kernel void kernel_mul_mv_q1_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q1_0_f32_impl<N_R0_Q1_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q2_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK2_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q2_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_q2_0 *) ((device char *) src0 + offset0);
    }

    float yl[16];
    float sumf[nr0] = {0.f};

    // group 64: 4 sub-blocks of 16 weights per Q2_0 block
    const short ix = (tiisg/4);
    const short il = (tiisg%4)*16;

    device const float * yb = y + ix*QK2_0 + il;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
        float sumy = 0.f;

        FOR_UNROLL (short i = 0; i < 16; i++) {
            yl[i] = yb[i];
            sumy += yb[i];
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy, yl, il);
        }

        yb += QK2_0 * (N_SIMDWIDTH/4);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q2_0_f32")]]
kernel void kernel_mul_mv_q2_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_0_f32_impl<N_R0_Q2_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q4_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q4_1_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
     mul_vec_q_n_f32_impl<block_q4_1, N_R0_Q4_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q5_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q5_1_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<short NR0, typename args_t>
void kernel_mul_mv_q8_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;

    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const block_q8_0 * x = (device const block_q8_0 *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NQ);
    const short il = tiisg%(NW/NQ);

    const int ib0 = sgitg*NQ + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    // each thread in a SIMD group deals with NQ quants at a time
    for (int ib = ib0; ib < nb; ib += NSG*NQ) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NQ*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// One SIMDgroup handles four HC streams for one output dimension and several
// tokens. Only the reduced result is written; gate activations stay in registers.
template<short COLS>
kernel void kernel_mul_mv_hc_q8(
        constant ggml_metal_kargs_mul_mv_ext & a,
        device const char * weights, device const char * input, device float * output,
        device const float * residual,
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int dim=group.x*2+sg;
    const int embd=a.ne01/4;
    if (dim>=embd) return;
    const int stream=lane/8;
    const int part=lane%8;
    const int c0=group.y*COLS;
    device const block_q8_0 * w=(device const block_q8_0 *)(weights+(dim+stream*embd)*a.nb01);
    float sums[COLS]={};
    for (int chunk=part;chunk<a.ne00/4;chunk+=8) {
        float4 q;
        dequantize_q8_0_t4(w+chunk/8,chunk%8,q);
        for (short c=0;c<COLS;++c) {
            device const float4 * x=(device const float4 *)(input+min(c0+c,a.ne11-1)*a.nb11);
            sums[c]+=dot(q,x[chunk]);
        }
    }
    for (short c=0;c<COLS;++c) {
        float z=sums[c];
        z+=simd_shuffle_down(z,4);
        z+=simd_shuffle_down(z,2);
        z+=simd_shuffle_down(z,1);
        const int column=min(c0+c,a.ne11-1);
        const float gate=1.0f/(1.0f+exp(-z));
        const float v=part==0 ? residual[column*a.ne01+stream*embd+dim]*gate : 0.0f;
        const float s1=simd_shuffle(v,8);
        const float s2=simd_shuffle(v,16);
        const float s3=simd_shuffle(v,24);
        if (lane==0 && c0+c<a.ne11) output[(c0+c)*embd+dim]=((v+s1)+s2+s3)*0.25f;
    }
}
typedef decltype(kernel_mul_mv_hc_q8<1>) mul_mv_hc_q8_t;
#define HC_UP_INST(C) template [[host_name("kernel_mul_mv_hc_q8_r1_" #C)]] kernel mul_mv_hc_q8_t kernel_mul_mv_hc_q8<C>;
HC_UP_INST(1) HC_UP_INST(2) HC_UP_INST(3) HC_UP_INST(4) HC_UP_INST(5)
#undef HC_UP_INST

// Small-batch Q8 experts: each SIMDgroup owns independent output rows.
// No shared-memory reduction, threadgroup barrier, or inter-group scratch.
template<short ROWS>
kernel void kernel_mul_mv_id_q8_rows(
        constant ggml_metal_kargs_mul_mv_id & a,
        device const char * weights, device const char * input,
        device char * output, device const char * ids,
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int token = int(group.z)/a.nei0;
    const int slot = int(group.z)%a.nei0;
    const int expert = ((device const int *)(ids + uint64_t(token)*a.nbi1))[slot];
    const int row0 = (int(group.x)*2 + int(sg))*ROWS;
    if (row0 >= a.ne01 || expert < 0 || expert >= a.ne02) return;
    device const float * x = (device const float *)(input + uint64_t(slot%a.ne11)*a.nb11 + uint64_t(token)*a.nb12);
    device const block_q8_0 * w[ROWS];
    FOR_UNROLL(short r=0; r<ROWS; ++r) {
        w[r] = (device const block_q8_0 *)(weights + uint64_t(expert)*a.nb02 + uint64_t(min(row0+r,a.ne01-1))*a.nb01);
    }
    float sum[ROWS] = {};
    const int qi = (lane%4)*8;
    for (int b=lane/4; b<a.ne00/32; b+=8) {
        const float4 x0 = float4(*((device const packed_float4 *)(x+b*32+qi)));
        const float4 x1 = float4(*((device const packed_float4 *)(x+b*32+qi+4)));
        FOR_UNROLL(short r=0; r<ROWS; ++r) {
            const float4 q0 = float4(*((device const packed_char4 *)(w[r][b].qs+qi)));
            const float4 q1 = float4(*((device const packed_char4 *)(w[r][b].qs+qi+4)));
            float partial = 0.0f;
            FOR_UNROLL(short j=0; j<4; ++j) partial += q0[j]*x0[j];
            FOR_UNROLL(short j=0; j<4; ++j) partial += q1[j]*x1[j];
            sum[r] += float(w[r][b].d)*partial;
        }
    }
    device float * y = (device float *)output + (uint64_t(token)*a.ne1+slot)*a.ne0;
    FOR_UNROLL(short r=0; r<ROWS; ++r) {
        const float value = simd_sum(sum[r]);
        if (lane==0 && row0+r<a.ne01) y[row0+r] = value;
    }
}
typedef decltype(kernel_mul_mv_id_q8_rows<2>) q8_expert_rows_t;
template [[host_name("kernel_mul_mv_id_q8_rows2")]] kernel q8_expert_rows_t kernel_mul_mv_id_q8_rows<2>;
template [[host_name("kernel_mul_mv_id_q8_rows4")]] kernel q8_expert_rows_t kernel_mul_mv_id_q8_rows<4>;
template [[host_name("kernel_mul_mv_id_q8_rows8")]] kernel q8_expert_rows_t kernel_mul_mv_id_q8_rows<8>;
template [[host_name("kernel_mul_mv_id_q8_rows16")]] kernel q8_expert_rows_t kernel_mul_mv_id_q8_rows<16>;
template [[host_name("kernel_mul_mv_id_q8_rows32")]] kernel q8_expert_rows_t kernel_mul_mv_id_q8_rows<32>;

// Block-dot experiment: scale each eight-element dot once and reuse activations
// across rows and quantized weights across columns. Each SIMDgroup is independent.
template<short ROWS, short COLS, typename QT=block_q8_0, bool LOW_REG=false, bool SOA=false>
kernel void kernel_mul_mv_q8_block(
        constant ggml_metal_kargs_mul_mv_ext & a,
        device const char * weights, device const char * input, device char * output,
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const bool pair_cols = FC_mul_mv_cache_tile == -1;
    const int row_sg = pair_cols ? FC_mul_mv_nsg/2 : FC_mul_mv_nsg;
    int row_group = group.x;
    int col_group = group.y;
    if (FC_mul_mv_cache_tile > 0) {
        const int nr = (a.ne01 + ROWS*row_sg - 1)/(ROWS*row_sg);
        const int nc = (a.ne11 + COLS - 1)/COLS;
        const int flat = int(group.x) + int(group.y)*nr;
        const int tile = FC_mul_mv_cache_tile;
        const int rb = (flat/(tile*nc))*tile;
        const int rows_here = min(tile, nr-rb);
        const int within = flat%(tile*nc);
        row_group = rb + within%rows_here;
        col_group = within/rows_here;
    }
    const int r0 = (row_group*row_sg + sg%row_sg)*ROWS;
    const int c0 = (pair_cols ? col_group*2 + sg/row_sg : col_group)*COLS;
    const int im = group.z;
    const int i12 = im%FC_mul_mv_ne12;
    const int i13 = im/FC_mul_mv_ne12;
    const uint64_t wa = (i12/FC_mul_mv_r2)*a.nb02 + (i13/FC_mul_mv_r3)*a.nb03;
    const uint64_t xb = i12*a.nb12 + i13*a.nb13;
    const int bi = lane/4;
    const int qi = (lane%4)*8;
    device const half * soa_d=(device const half *)(weights+(uint64_t)a.ne00*a.ne01);
    device const char * soa_q[ROWS];
    device const QT * wr[ROWS];
    for (short r=0; r<ROWS; ++r) {
        wr[r] = (device const QT *)(weights + wa + min(r0+r, a.ne01-1)*a.nb01);
        soa_q[r]=weights+(uint64_t)min(r0+r,a.ne01-1)*a.ne00;
    }
    device const float * xc[COLS];
    for (short c=0; c<COLS; ++c) {
        xc[c] = (device const float *)(input + xb + min(c0+c, a.ne11-1)*a.nb11);
    }
    float sum[ROWS][COLS] = {};
    for (int b=bi; b<a.ne00/32; b+=8) {
        if (FC_mul_mv_vector_load > 0) {
            float4 x0[COLS], x1[COLS];
            FOR_UNROLL(short c=0;c<COLS;++c) {
                x0[c]=float4(*((device const packed_float4 *)(xc[c]+b*32+qi)));
                x1[c]=float4(*((device const packed_float4 *)(xc[c]+b*32+qi+4)));
            }
            FOR_UNROLL(short r=0;r<ROWS;++r) {
                float4 q0,q1;
                if(is_same<QT,block_q4_0>::value) {
                    const ushort shift=qi>=16 ? 4 : 0;
                    const uchar4 a0=uchar4(*((device const packed_uchar4 *)(wr[r][b].qs+qi%16)));
                    const uchar4 a1=uchar4(*((device const packed_uchar4 *)(wr[r][b].qs+qi%16+4)));
                    q0=float4(int4((a0>>shift)&uchar4(15))-int4(8));
                    q1=float4(int4((a1>>shift)&uchar4(15))-int4(8));
                } else {
                    q0=float4(*((device const packed_char4 *)(SOA ? soa_q[r]+b*32+qi : (device const char *)wr[r][b].qs+qi)));
                    q1=float4(*((device const packed_char4 *)(SOA ? soa_q[r]+b*32+qi+4 : (device const char *)wr[r][b].qs+qi+4)));
                }
                const float d=(SOA ? float(soa_d[(uint64_t)min(r0+r,a.ne01-1)*(a.ne00/32)+b]) : float(wr[r][b].d));
                FOR_UNROLL(short c=0;c<COLS;++c) {
                    float value=0.0f;
                    if (FC_mul_mv_vector_load == 1) {
                        FOR_UNROLL(short j=0;j<4;++j) value+=q0[j]*x0[c][j];
                        FOR_UNROLL(short j=0;j<4;++j) value+=q1[j]*x1[c][j];
                    } else {
                        value=dot(q0,x0[c])+dot(q1,x1[c]);
                    }
                    sum[r][c]+=d*value;
                }
            }
        } else if (LOW_REG) {
            float q[ROWS][8];
            float scale[ROWS];
#pragma unroll
            for (short r=0; r<ROWS; ++r) {
                scale[r]=(SOA ? float(soa_d[(uint64_t)min(r0+r,a.ne01-1)*(a.ne00/32)+b]) : float(wr[r][b].d));
                FOR_UNROLL(short j=0; j<8; ++j) {
                    if (is_same<QT,block_q4_0>::value) {
                        const int k=qi+j;
                        q[r][j]=float(int((wr[r][b].qs[k%16]>>(k>=16?4:0))&15)-8);
                    } else q[r][j]=float(SOA ? soa_q[r][b*32+qi+j] : wr[r][b].qs[qi+j]);
                }
            }
#pragma unroll
            for (short c=0; c<COLS; ++c) {
                float x[8];
                FOR_UNROLL(short j=0;j<8;++j) x[j]=xc[c][b*32+qi+j];
#pragma unroll
                for(short r=0;r<ROWS;++r) {
                    float dotq=0.0f;
                    FOR_UNROLL(short j=0;j<8;++j) dotq+=q[r][j]*x[j];
                    sum[r][c]+=scale[r]*dotq;
                }
            }
        } else {
        float x[COLS][8];
        for (short c=0; c<COLS; ++c) {
            FOR_UNROLL (short j=0; j<8; ++j) x[c][j] = xc[c][b*32+qi+j];
        }
        for (short r=0; r<ROWS; ++r) {
            float q[8];
            FOR_UNROLL (short j=0; j<8; ++j) {
                if (is_same<QT,block_q4_0>::value) {
                    const int k=qi+j;
                    q[j]=float(int((wr[r][b].qs[k%16]>>(k>=16?4:0))&15)-8);
                } else {
                    q[j] = float(SOA ? soa_q[r][b*32+qi+j] : wr[r][b].qs[qi+j]);
                }
            }
            const float d = (SOA ? float(soa_d[(uint64_t)min(r0+r,a.ne01-1)*(a.ne00/32)+b]) : float(wr[r][b].d));
            for (short c=0; c<COLS; ++c) {
                float dotq=0.0f;
                FOR_UNROLL (short j=0; j<8; ++j) dotq += q[j]*x[c][j];
                sum[r][c] += d*dotq;
            }
        }
        }
    }
    for (short r=0; r<ROWS; ++r) {
        for (short c=0; c<COLS; ++c) {
            float v=simd_sum(sum[r][c]);
            if (lane==0 && r0+r<a.ne01 && c0+c<a.ne11) {
                ((device float *)output)[(uint64_t)im*a.ne0*a.ne1 + (uint64_t)(c0+c)*a.ne0 + r0+r]=v;
            }
        }
    }
}
typedef decltype(kernel_mul_mv_q8_block<2,2>) mul_mv_q8_block_t;
#define Q8_BLOCK_INST(R,C) template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_" #C "_b" #R)]] kernel mul_mv_q8_block_t kernel_mul_mv_q8_block<R,C>;
Q8_BLOCK_INST(1,2) Q8_BLOCK_INST(1,3) Q8_BLOCK_INST(1,4) Q8_BLOCK_INST(1,5)
Q8_BLOCK_INST(2,2) Q8_BLOCK_INST(2,3) Q8_BLOCK_INST(2,4) Q8_BLOCK_INST(2,5)
Q8_BLOCK_INST(4,2) Q8_BLOCK_INST(4,3) Q8_BLOCK_INST(4,4) Q8_BLOCK_INST(4,5)
Q8_BLOCK_INST(8,2) Q8_BLOCK_INST(8,3) Q8_BLOCK_INST(8,4) Q8_BLOCK_INST(8,5)
Q8_BLOCK_INST(1,6) Q8_BLOCK_INST(1,7) Q8_BLOCK_INST(1,8)
Q8_BLOCK_INST(2,6) Q8_BLOCK_INST(2,7) Q8_BLOCK_INST(2,8)
Q8_BLOCK_INST(4,6) Q8_BLOCK_INST(4,7) Q8_BLOCK_INST(4,8)
Q8_BLOCK_INST(8,6) Q8_BLOCK_INST(8,7) Q8_BLOCK_INST(8,8)
#undef Q8_BLOCK_INST

#define H90_SOA_INST(R,C) template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_" #C "_a" #R)]] kernel mul_mv_q8_block_t kernel_mul_mv_q8_block<R,C,block_q8_0,false,true>;
H90_SOA_INST(1,2) H90_SOA_INST(1,3) H90_SOA_INST(1,4) H90_SOA_INST(1,5) H90_SOA_INST(1,6) H90_SOA_INST(1,7) H90_SOA_INST(1,8)
H90_SOA_INST(2,2) H90_SOA_INST(2,3) H90_SOA_INST(2,4) H90_SOA_INST(2,5) H90_SOA_INST(2,6) H90_SOA_INST(2,7) H90_SOA_INST(2,8)
H90_SOA_INST(4,2) H90_SOA_INST(4,3) H90_SOA_INST(4,4) H90_SOA_INST(4,5) H90_SOA_INST(4,6) H90_SOA_INST(4,7) H90_SOA_INST(4,8)
H90_SOA_INST(8,2) H90_SOA_INST(8,3) H90_SOA_INST(8,4) H90_SOA_INST(8,5) H90_SOA_INST(8,6) H90_SOA_INST(8,7) H90_SOA_INST(8,8)
#undef H90_SOA_INST

#define Q4_BLOCK_INST(R,C) template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_" #C "_b" #R)]] kernel mul_mv_q8_block_t kernel_mul_mv_q8_block<R,C,block_q4_0>;
Q4_BLOCK_INST(1,2) Q4_BLOCK_INST(1,3) Q4_BLOCK_INST(1,4) Q4_BLOCK_INST(1,5) Q4_BLOCK_INST(1,6) Q4_BLOCK_INST(1,7) Q4_BLOCK_INST(1,8)
Q4_BLOCK_INST(2,2) Q4_BLOCK_INST(2,3) Q4_BLOCK_INST(2,4) Q4_BLOCK_INST(2,5) Q4_BLOCK_INST(2,6) Q4_BLOCK_INST(2,7) Q4_BLOCK_INST(2,8)
Q4_BLOCK_INST(4,2) Q4_BLOCK_INST(4,3) Q4_BLOCK_INST(4,4) Q4_BLOCK_INST(4,5) Q4_BLOCK_INST(4,6) Q4_BLOCK_INST(4,7) Q4_BLOCK_INST(4,8)
Q4_BLOCK_INST(8,2) Q4_BLOCK_INST(8,3) Q4_BLOCK_INST(8,4) Q4_BLOCK_INST(8,5) Q4_BLOCK_INST(8,6) Q4_BLOCK_INST(8,7) Q4_BLOCK_INST(8,8)
#undef Q4_BLOCK_INST
#define SLIM_BLOCK_INST(Q,T,R,C) template [[host_name("kernel_mul_mv_ext_" #Q "_f32_r1_" #C "_s" #R)]] kernel mul_mv_q8_block_t kernel_mul_mv_q8_block<R,C,T,true>;
#define SLIM_BLOCK_ROW(Q,T,R) SLIM_BLOCK_INST(Q,T,R,2) SLIM_BLOCK_INST(Q,T,R,3) SLIM_BLOCK_INST(Q,T,R,4) SLIM_BLOCK_INST(Q,T,R,5) SLIM_BLOCK_INST(Q,T,R,6) SLIM_BLOCK_INST(Q,T,R,7) SLIM_BLOCK_INST(Q,T,R,8)
SLIM_BLOCK_ROW(q8_0,block_q8_0,1) SLIM_BLOCK_ROW(q8_0,block_q8_0,2) SLIM_BLOCK_ROW(q8_0,block_q8_0,4) SLIM_BLOCK_ROW(q8_0,block_q8_0,8)
SLIM_BLOCK_ROW(q4_0,block_q4_0,1) SLIM_BLOCK_ROW(q4_0,block_q4_0,2) SLIM_BLOCK_ROW(q4_0,block_q4_0,4) SLIM_BLOCK_ROW(q4_0,block_q4_0,8)
SLIM_BLOCK_INST(q8_0,block_q8_0,2,12)
SLIM_BLOCK_INST(q8_0,block_q8_0,2,16)
#undef SLIM_BLOCK_ROW
#undef SLIM_BLOCK_INST

// mat-vec kernel processing in chunks of float4
// chpb - chunks per quantization block
template<short r1ptg, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &) >
void kernel_mul_mv_ext_q4_f32_impl(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 4; // chunks per thread

  //const short nxpsg = (32);
    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%FC_mul_mv_ne12;
    const int i13 = i1m/FC_mul_mv_ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb; // current chunk index

    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    // reduce only the threads in each row
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }

        //sumf[ir1] = simd_sum(sumf[ir1]);
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// mat-vec kernel processing in chunks of float4x4
template<short r1ptg, typename q_t, short chpb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &) >
void kernel_mul_mv_ext_q4x4_f32_impl(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 1;

  //const short nxpsg = (32);
    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%FC_mul_mv_ne12;
    const int i13 = i1m/FC_mul_mv_ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4x4 * y4x4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4x4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4x4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4x4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb;

    for (int ich = tx; 16*ich < args.ne00; ich += chpt*nxpsg) {
        float4x4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4x4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] +=
                    dot(lx[ch][0], y4x4[ir1][ch*nxpsg][0]) +
                    dot(lx[ch][1], y4x4[ir1][ch*nxpsg][1]) +
                    dot(lx[ch][2], y4x4[ir1][ch*nxpsg][2]) +
                    dot(lx[ch][3], y4x4[ir1][ch*nxpsg][3]);

            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4x4[ir1] += chpt*nxpsg;
        }
    }

    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }

        //sumf[ir1] = simd_sum(sumf[ir1]);
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// dispatchers needed for compile-time nxpsg
// epb - elements per quantization block
template<short r1ptg, typename q_t, short epb, void (*deq_t4)(device const q_t *, short, thread float4 &)>
kernel void kernel_mul_mv_ext_q4_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4_f32_impl<r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

template<short r1ptg, typename q_t, short epb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &)>
kernel void kernel_mul_mv_ext_q4x4_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4x4_f32_impl<r1ptg, q_t, epb/16, deq_t4x4>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_ext_q4_f32_disp  <2, block_q8_0, 32,  dequantize_q8_0_t4>) mul_mv_ext_q4_f32_t;
typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, block_q4_K, 256, dequantize_q4_K>)    mul_mv_ext_q4x4_f32_t;

template [[host_name("kernel_mul_mv_ext_f32_f32_r1_2")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, float4,       4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_3")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, float4,       4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_4")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, float4,       4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_5")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, float4,       4,  dequantize_f32_t4>;

template [[host_name("kernel_mul_mv_ext_f16_f32_r1_2")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_3")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_4")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_5")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, half4,        4,  dequantize_f16_t4>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_ext_bf16_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, bfloat4,      4,  dequantize_bf16_t4>;
template [[host_name("kernel_mul_mv_ext_bf16_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, bfloat4,      4,  dequantize_bf16_t4>;
template [[host_name("kernel_mul_mv_ext_bf16_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, bfloat4,      4,  dequantize_bf16_t4>;
template [[host_name("kernel_mul_mv_ext_bf16_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, bfloat4,      4,  dequantize_bf16_t4>;
#endif

template [[host_name("kernel_mul_mv_ext_q1_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q1_0,   128, dequantize_q1_0_t4>;
template [[host_name("kernel_mul_mv_ext_q1_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q1_0,   128, dequantize_q1_0_t4>;
template [[host_name("kernel_mul_mv_ext_q1_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q1_0,   128, dequantize_q1_0_t4>;
template [[host_name("kernel_mul_mv_ext_q1_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q1_0,   128, dequantize_q1_0_t4>;

template [[host_name("kernel_mul_mv_ext_q2_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q2_0,    64, dequantize_q2_0_t4>;
template [[host_name("kernel_mul_mv_ext_q2_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q2_0,    64, dequantize_q2_0_t4>;
template [[host_name("kernel_mul_mv_ext_q2_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q2_0,    64, dequantize_q2_0_t4>;
template [[host_name("kernel_mul_mv_ext_q2_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q2_0,    64, dequantize_q2_0_t4>;

template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q4_0,   32, dequantize_q4_0_t4>;

template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q4_1,   32, dequantize_q4_1_t4>;

template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q5_0,   32, dequantize_q5_0_t4>;

template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q5_1,   32, dequantize_q5_1_t4>;

template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q8_0,   32, dequantize_q8_0_t4>;

template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_2")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_mxfp4,  32, dequantize_mxfp4_t4>;
template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_3")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_mxfp4,  32, dequantize_mxfp4_t4>;
template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_4")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_mxfp4,  32, dequantize_mxfp4_t4>;
template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_5")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_mxfp4,  32, dequantize_mxfp4_t4>;

template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_iq4_nl, 32, dequantize_iq4_nl_t4>;

template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q4_K, 256, dequantize_q4_K>;

template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q5_K, 256, dequantize_q5_K>;

template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q6_K, 256, dequantize_q6_K>;

template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q2_K, 256, dequantize_q2_K>;

template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, block_q3_K, 256, dequantize_q3_K>;

template<typename T0, typename T1, short NR0, typename args_t>
void kernel_mul_mv_t_t_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB = 32;
    constexpr short NF = 8;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const T0 * x = (device const T0 *) (src0 + offset0);
    device const T1 * y = (device const T1 *) (src1 + offset1);

    // pointers to src0 rows
    device const T0 * ax [NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax[row] = (device const T0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T1 yl[NF];

    device const T1 * yb = y + (ib0*NB + il*NF);

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T0 * xb = ax[row] + (ib*NB + il*NF);

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF; ++i) {
                sumq += xb[i] * yl[i];
            }

            sumf[row] += sumq;
        }

        yb += NSG*NF*NW;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
      //case 1: kernel_mul_mv_t_t_impl<T0, T1, 1, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 2: kernel_mul_mv_t_t_impl<T0, T1, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
      //case 3: kernel_mul_mv_t_t_impl<T0, T1, 3, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
      //case 4: kernel_mul_mv_t_t_impl<T0, T1, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    }
}

template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_disp<T0, T1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t<half, half>) mul_mv_t_t;

template [[host_name("kernel_mul_mv_f32_f32")]]   kernel mul_mv_t_t kernel_mul_mv_t_t<float, float>;
template [[host_name("kernel_mul_mv_f16_f32")]]   kernel mul_mv_t_t kernel_mul_mv_t_t<half,  float>;
template [[host_name("kernel_mul_mv_f16_f16")]]   kernel mul_mv_t_t kernel_mul_mv_t_t<half,  half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_f32")]]  kernel mul_mv_t_t kernel_mul_mv_t_t<bfloat, float>;
template [[host_name("kernel_mul_mv_bf16_bf16")]] kernel mul_mv_t_t kernel_mul_mv_t_t<bfloat, bfloat>;
#endif

template<typename T0, typename T04, typename T1, typename T14, short NR0, typename args_t>
void kernel_mul_mv_t_t_4_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr short NW = N_SIMDWIDTH;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const T1  * y  = (device const T1  *) (src1 + offset1);
    device const T14 * y4 = (device const T14 *) (src1 + offset1);

    // pointers to src0 rows
    device const T0  * ax [NR0];
    device const T04 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax [row] = (device const T0  *) ((device char *) src0 + offset0);
        ax4[row] = (device const T04 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);

    const int ib0 = sgitg*NF + ix;

    T14 yl4[NF4];

    device const T14 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T04 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), float4(yl4[i]));
            }

            sumf[row] += sumq;
        }

        yb4 += NSG*NF*NW/4;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T04, typename T1, typename T14, typename args_t>
void kernel_mul_mv_t_t_4_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
        case 1: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 1, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 2: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
      //case 3: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 3, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
      //case 4: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    };
}

template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv_t_t_4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_4_disp<T0, T04, T1, T14, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t_4<half, half4, half, half4>) mul_mv_t_t_4;

template [[host_name("kernel_mul_mv_f32_f32_4")]]   kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<float, float4, float, float4>;
template [[host_name("kernel_mul_mv_f16_f32_4")]]   kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<half,  half4,  float, float4>;
template [[host_name("kernel_mul_mv_f16_f16_4")]]   kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<half,  half4,  half,  half4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_f32_4")]]  kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<bfloat, bfloat4, float,  float4>;
template [[host_name("kernel_mul_mv_bf16_bf16_4")]] kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<bfloat, bfloat4, bfloat, bfloat4>;
#endif

template<typename T0, typename T04>
kernel void kernel_mul_mv_t_t_4_nr4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_4_impl<T0, T04, float, float4, 4, constant ggml_metal_kargs_mul_mv &>(
        args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t_4_nr4<half, half4>) mul_mv_t_t_4_nr4;

template [[host_name("kernel_mul_mv_f16_f32_4_nr4")]] kernel mul_mv_t_t_4_nr4 kernel_mul_mv_t_t_4_nr4<half, half4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_f32_4_nr4")]] kernel mul_mv_t_t_4_nr4 kernel_mul_mv_t_t_4_nr4<bfloat, bfloat4>;
#endif

template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_short_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg) {
    const int r0 = tgpig.x*32 + tiisg;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    if (r0 >= args.ne01) {
        return;
    }

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

    device const T0 * x = (device const T0 *) (src0 + offset0);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;

    const uint64_t offset1 = r1*args.nb11 + (i12   )*args.nb12 + (i13   )*args.nb13;

    device const T1 * y = (device const T1 *) (src1 + offset1);

    float res = 0.0f;

    for (int i = 0; i < args.ne00; ++i) {
        res += (float) x[i] * (float) y[i];
    }

    dst_f32[(uint64_t)r1*args.ne0 + r0] = res;
}

template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t_short(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_t_t_short_impl<T0, T1, constant ggml_metal_kargs_mul_mv &>(
        args,
        src0,
        src1,
        dst,
        tgpig,
        tiisg);
}

typedef decltype(kernel_mul_mv_t_t_short<half, half>) mul_mv_t_t_short_t;

template [[host_name("kernel_mul_mv_f32_f32_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<float, float>;
template [[host_name("kernel_mul_mv_f16_f32_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<half,  float>;
template [[host_name("kernel_mul_mv_f16_f16_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<half,  half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_f32_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<bfloat, float>;
template [[host_name("kernel_mul_mv_bf16_bf16_short")]] kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<bfloat, bfloat>;
#endif

template<int nr0, typename args_t>
void kernel_mul_mv_q2_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3
    const short is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + 8*iq + is;
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qs[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qs[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qs[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qs[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qs[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qs[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qs[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qs[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += args.nb01/2;
            sc += args.nb01;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q2_K_f32")]]
kernel void kernel_mul_mv_q2_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q3_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q3_K * x = (device const block_q3_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float yl[32];

    //const uint16_t kmask1 = 0x3030;
    //const uint16_t kmask2 = 0x0f0f;

    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short ip  = tid/4;          // 0 or 1
    const short il  = 2*((tid%4)/2);  // 0 or 2
    const short ir  = tid%2;
    const short l0  = 8*ir;

    // One would think that the Metal compiler would figure out that ip and il can only have
    // 4 possible states, and optimize accordingly. Well, no. It needs help, and we do it
    // with these two tales.
    //
    // Possible masks for the high bit
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},  // ip = 0, il = 0
                           {0x0004, 0x0400, 0x0008, 0x0800},  // ip = 0, il = 2
                           {0x0010, 0x1000, 0x0020, 0x2000},  // ip = 1, il = 0
                           {0x0040, 0x4000, 0x0080, 0x8000}}; // ip = 1, il = 2

    // Possible masks for the low 2 bits
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};

    const ushort4 hm = mm[2*ip + il/2];

    const short shift = 2*il;

    const float v1 = il == 0 ? 4.f : 64.f;
    const float v2 = 4.f * v1;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;

    const short q_offset = 32*ip + l0;
    const short y_offset = 128*ip + 32*il + l0;

    device const float * y1 = yy + ix*QK_K + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[nr0] = {0.f};
    float sumf2[nr0] = {0.f};

    for (int i = ix; i < nb; i += 4) {
        for (short l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        device const uint16_t * q = (device const uint16_t *)(x[i].qs + q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;

        for (short row = 0; row < nr0; ++row) {
            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2];
                s1 += yl[l+0] * (qs & qm[il/2][0]);
                s2 += yl[l+1] * (qs & qm[il/2][1]);
                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm[il/2][2]);
                s5 += yl[l+17] * (qs & qm[il/2][3]);
                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2+8];
                s1 += yl[l+8] * (qs & qm[il/2][0]);
                s2 += yl[l+9] * (qs & qm[il/2][1]);
                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qs & qm[il/2][2]);
                s5 += yl[l+25] * (qs & qm[il/2][3]);
                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);

            q  += args.nb01/2;
            h  += args.nb01/2;
            a  += args.nb01/2;
            dh += args.nb01/2;
        }

        y1 += 4 * QK_K;
    }

    for (int row = 0; row < nr0; ++row) {
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sumf);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (tiisg == 0) {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            dst_f32[first_row + row] = sumf1[row];
        }
    }
}

[[host_name("kernel_mul_mv_q3_K_f32")]]
kernel void kernel_mul_mv_q3_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// Affine block dot: each lane handles eight values; scale and offset are
// applied once per partial block and the input is shared across output rows.
template<short ROWS>
kernel void kernel_mul_mv_id_q5_block(
        constant ggml_metal_kargs_mul_mv_id & a,
        device const char * weights, device const char * input, device char * output,
        device const char * ids, uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int row0=(group.x*FC_mul_mv_nsg+sg)*ROWS;
    const int token=group.z/a.nei0;
    const int slot=group.z%a.nei0;
    const int expert=((device const int *)(ids+token*a.nbi1))[slot];
    device const float * y=(device const float *)(input+token*a.nb12+(slot%a.ne11)*a.nb11);
    device const block_q5_1 * w[ROWS];
    for (short r=0;r<ROWS;++r) {
        w[r]=(device const block_q5_1 *)(weights+expert*a.nb02+min(row0+r,a.ne01-1)*a.nb01);
    }
    const int qi=(lane%4)*8;
    float sums[ROWS]={};
    for (int b=lane/4;b<a.ne00/32;b+=8) {
        float yy[8];
        float sy=0.0f;
        FOR_UNROLL (short j=0;j<8;++j) { yy[j]=y[b*32+qi+j]; sy+=yy[j]; }
        for (short r=0;r<ROWS;++r) {
            const uint high=*((device const uint *)w[r][b].qh);
            float dotq=0.0f;
            FOR_UNROLL (short j=0;j<8;++j) {
                const int q=qi+j;
                const uint lo=(uint(w[r][b].qs[q%16])>>(4*(q/16)))&15;
                const uint hi=((high>>q)&1)<<4;
                dotq+=float(lo|hi)*yy[j];
            }
            sums[r]+=float(w[r][b].d)*dotq+float(w[r][b].m)*sy;
        }
    }
    for (short r=0;r<ROWS;++r) {
        const float value=simd_sum(sums[r]);
        if (lane==0 && row0+r<a.ne01) {
            ((device float *)(output+(uint64_t)(token*a.ne1+slot)*a.nb1))[row0+r]=value;
        }
    }
}
typedef decltype(kernel_mul_mv_id_q5_block<2>) mul_mv_id_q5_block_t;
#define Q5_BLOCK_INST(R) template [[host_name("kernel_mul_mv_id_q5_1_f32_b" #R)]] kernel mul_mv_id_q5_block_t kernel_mul_mv_id_q5_block<R>;
Q5_BLOCK_INST(1) Q5_BLOCK_INST(2) Q5_BLOCK_INST(4) Q5_BLOCK_INST(8)
#undef Q5_BLOCK_INST

// Short-K expert projection: map independent rows to subgroups and load four
// contiguous quantized elements per lane. No threadgroup-memory staging.
template<short NX, short ROWS>
kernel void kernel_mul_mv_id_q5_short(
        constant ggml_metal_kargs_mul_mv_id & a,
        device const char * weights, device const char * input, device char * output,
        device const char * ids, uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NY=32/NX;
    const int row0=((group.x*FC_mul_mv_nsg+sg)*NY+lane/NX)*ROWS;
    const int tx=lane%NX;
    const int token=group.z/a.nei0, slot=group.z%a.nei0;
    const int expert=((device const int *)(ids+token*a.nbi1))[slot];
    device const float4 * x=(device const float4 *)(input+token*a.nb12+(slot%a.ne11)*a.nb11);
    float sums[ROWS]={};
    for(int c=tx;c<a.ne00/4;c+=NX) {
        const float4 v=x[c];
        const float vs=v[0]+v[1]+v[2]+v[3];
        const int b=c/8, q=(c%8)*4;
        for(short r=0;r<ROWS;++r) {
            device const block_q5_1 * w=(device const block_q5_1 *)(weights+expert*a.nb02+min(row0+r,a.ne01-1)*a.nb01)+b;
            uint h=*((device const uint *)w->qh);
            uint packed=*((device const uint *)(w->qs+(q%16)));
            const uint shift=q>=16?4:0;
            float4 values;
            FOR_UNROLL(short j=0;j<4;++j) values[j]=float(((packed>>(8*j+shift))&15u)|(((h>>(q+j))&1u)<<4));
            sums[r]+=float(w->d)*dot(values,v)+float(w->m)*vs;
        }
    }
    for(short r=0;r<ROWS;++r) {
        for(short off=NX/2;off>0;off/=2) sums[r]+=simd_shuffle_down(sums[r],off);
        if(tx==0 && row0+r<a.ne01) {
            ((device float *)(output+(token*a.ne1+slot)*a.nb1))[row0+r]=sums[r];
        }
    }
}
typedef decltype(kernel_mul_mv_id_q5_short<8,1>) mul_mv_id_q5_short_t;
#define Q5_SHORT_INST(X,R) template [[host_name("kernel_mul_mv_id_q5_1_f32_s" #X "r" #R)]] kernel mul_mv_id_q5_short_t kernel_mul_mv_id_q5_short<X,R>;
Q5_SHORT_INST(4,1) Q5_SHORT_INST(4,2) Q5_SHORT_INST(8,1) Q5_SHORT_INST(8,2)
Q5_SHORT_INST(16,1) Q5_SHORT_INST(16,2) Q5_SHORT_INST(32,1) Q5_SHORT_INST(32,2)
#undef Q5_SHORT_INST

// Pair repeated expert selections in occurrence order. Two columns bound the
// accumulator footprint; singleton selections use the same packed Q4 dot.
kernel void kernel_mul_mv_id_pair_map(
        constant ggml_metal_kargs_mul_mv_id & a, device const char * ids,
        device int * pairs, uint p [[thread_position_in_grid]]) {
    if (p >= uint(a.nei0*a.nei1)) return;
    const int expert=((device const int *)(ids+(p/a.nei0)*a.nbi1))[p%a.nei0];
    int before=0, next=-1;
    for (uint j=0;j<p;++j) before+=((device const int *)(ids+(j/a.nei0)*a.nbi1))[j%a.nei0]==expert;
    if (before&1) { pairs[p]=-2; return; }
    for (uint j=p+1;j<uint(a.nei0*a.nei1);++j) {
        if (((device const int *)(ids+(j/a.nei0)*a.nbi1))[j%a.nei0]==expert) { next=int(j); break; }
    }
    pairs[p]=next;
}

template<short ROWS, bool MAPPED=false>
kernel void kernel_mul_mv_id_q4_pair(
        constant ggml_metal_kargs_mul_mv_id & a,
        device const char * weights, device const char * input, device char * output,
        device const char * ids, device const int * pairs,
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int flat = group.z;
    const int expert = ((device const int *)(ids+(flat/a.nei0)*a.nbi1))[flat%a.nei0];
    int previous=0, second=-1;
    if (MAPPED) { second=pairs[flat]; previous=second==-2; }
    else if (lane==0) {
        for (int p=0; p<flat; ++p) {
            const int e=((device const int *)(ids+(p/a.nei0)*a.nbi1))[p%a.nei0];
            previous += e==expert;
        }
        if ((previous&1)==0) {
            for (int p=flat+1; p<a.nei0*a.nei1; ++p) {
                const int e=((device const int *)(ids+(p/a.nei0)*a.nbi1))[p%a.nei0];
                if (e==expert) { second=p; break; }
            }
        }
    }
    previous=simd_shuffle(previous,0);
    second=simd_shuffle(second,0);
    if (previous&1) return;
    const int first_row=(group.x*FC_mul_mv_nsg+sg)*ROWS;
    const short ix=lane/8, iq=(lane%8)/4, ir=lane%4;
    const int outputs[2]={flat,second};
    device const float * y[2];
    for (short c=0; c<2; ++c) {
        const int p=max(outputs[c],0);
        y[c]=(device const float *)(input+(p/a.nei0)*a.nb12+((p%a.nei0)%a.ne11)*a.nb11);
    }
    float sums[ROWS][2]={};
    for (int ib=ix; ib<a.ne00/QK_K; ib+=4) {
        for (short row=0; row<ROWS; ++row) {
            device const block_q4_K * x=(device const block_q4_K *)(weights+expert*a.nb02+(first_row+row)*a.nb01)+ib;
            device const uint16_t * sc=(device const uint16_t *)x->scales+iq;
            uint16_t s16[4];
            s16[0]=sc[0]&0x3f3f; s16[1]=sc[2]&0x3f3f;
            s16[2]=(sc[4]&0x0f0f)|((sc[0]&0xc0c0)>>2);
            s16[3]=((sc[4]>>4)&0x0f0f)|((sc[2]&0xc0c0)>>2);
            thread const uint8_t * s8=(thread const uint8_t *)s16;
            device const uint16_t * qp=(device const uint16_t *)x->qs+16*iq+4*ir;
            uint16_t ql[4],qh[4];
            FOR_UNROLL(short j=0;j<4;++j) { ql[j]=qp[j]; qh[j]=qp[j+32]; }
            const float d=x->d, dm=x->dmin;
            for (short c=0; c<2; ++c) {
                if (outputs[c]<0) continue;
                device const float * yc=y[c]+ib*QK_K+64*iq+8*ir;
                float yl[16],yh[16];
                float4 sy=0.0f;
                FOR_UNROLL(short j=0;j<8;++j) {
                    yl[j]=yc[j]; sy[0]+=yl[j];
                    yl[j+8]=yc[j+32]; sy[1]+=yl[j+8];
                    yh[j]=yc[j+128]; sy[2]+=yh[j];
                    yh[j+8]=yc[j+160]; sy[3]+=yh[j+8];
                }
                float4 l=0.0f,h=0.0f;
                FOR_UNROLL(short j=0;j<4;++j) {
                    l[0]+=yl[2*j]*(ql[j]&0x000f);
                    l[1]+=yl[2*j+1]*(ql[j]&0x0f00);
                    l[2]+=yl[2*j+8]*(ql[j]&0x00f0);
                    l[3]+=yl[2*j+9]*(ql[j]&0xf000);
                    h[0]+=yh[2*j]*(qh[j]&0x000f);
                    h[1]+=yh[2*j+1]*(qh[j]&0x0f00);
                    h[2]+=yh[2*j+8]*(qh[j]&0x00f0);
                    h[3]+=yh[2*j+9]*(qh[j]&0xf000);
                }
                sums[row][c]+=d*((l[0]+(1.0f/256.0f)*l[1])*s8[0]+
                    (l[2]+(1.0f/256.0f)*l[3])*s8[1]*(1.0f/16.0f)+
                    (h[0]+(1.0f/256.0f)*h[1])*s8[4]+
                    (h[2]+(1.0f/256.0f)*h[3])*s8[5]*(1.0f/16.0f))-
                    dm*(sy[0]*s8[2]+sy[1]*s8[3]+sy[2]*s8[6]+sy[3]*s8[7]);
            }
        }
    }
    for (short row=0;row<ROWS;++row) {
        for(short c=0;c<2;++c) {
            const float v=simd_sum(sums[row][c]);
            if(lane==0 && outputs[c]>=0 && first_row+row<a.ne01) {
                const int p=outputs[c];
                device float * dst=(device float *)(output+((p/a.nei0)*a.ne1+p%a.nei0)*a.nb1);
                dst[first_row+row]=v;
            }
        }
    }
}
typedef decltype(kernel_mul_mv_id_q4_pair<2>) mul_mv_id_q4_pair_t;
template [[host_name("kernel_mul_mv_id_q4_K_f32_pair2")]] kernel mul_mv_id_q4_pair_t kernel_mul_mv_id_q4_pair<2>;
template [[host_name("kernel_mul_mv_id_q4_K_f32_pair4")]] kernel mul_mv_id_q4_pair_t kernel_mul_mv_id_q4_pair<4>;
template [[host_name("kernel_mul_mv_id_q4_K_f32_pairmap2")]] kernel mul_mv_id_q4_pair_t kernel_mul_mv_id_q4_pair<2,true>;
template [[host_name("kernel_mul_mv_id_q4_K_f32_pairmap4")]] kernel mul_mv_id_q4_pair_t kernel_mul_mv_id_q4_pair<4,true>;

// Exact packed-Q4 arithmetic for two projections sharing activations and IDs.
// Each SIMDgroup owns complete output rows; intermediate gates stay in registers.
template<short ROWS>
kernel void kernel_mul_mv_id_q4_gateup(
        constant ggml_metal_kargs_mul_mv_id & a,
        device const char * gates, device const char * ups,
        device const char * input, device const char * ids, device char * output,
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int row0 = (group.x*2 + sg)*ROWS;
    const int seq_col = group.z/a.nei0;
    const int slot = group.z%a.nei0;
    const int expert = ((device const int *)(ids + uint64_t(seq_col)*a.nbi1))[slot];
    device const float * y = (device const float *)(input + uint64_t(seq_col)*a.nb12 + (slot%a.ne11)*a.nb11);
    const short ix=lane/8, iq=(lane%8)/4, ir=lane%4;
    float sums[2*ROWS] = {};
    float yl[16], yh[16];
    ushort scales[4];
    thread const uchar * s = (thread const uchar *)scales;
    device const float * y4 = y + ix*256 + 64*iq + 8*ir;
    for (int ib=ix; ib<a.ne00/256; ib+=4) {
        float4 sy = 0.0f;
        for (short i=0; i<8; ++i) {
            yl[i]=y4[i]; yl[i+8]=y4[i+32];
            yh[i]=y4[i+128]; yh[i+8]=y4[i+160];
            sy[0]+=yl[i]; sy[1]+=yl[i+8]; sy[2]+=yh[i]; sy[3]+=yh[i+8];
        }
        for (short p=0; p<2; ++p) {
            device const char * base = p==0 ? gates : ups;
            for (short r=0; r<ROWS; ++r) {
                device const block_q4_K * w = (device const block_q4_K *)(base + uint64_t(expert)*a.nb02 + uint64_t(min(row0+r,a.ne01-1))*a.nb01) + ib;
                device const ushort * sc = (device const ushort *)w->scales + iq;
                device const ushort * q1 = (device const ushort *)w->qs + 16*iq + 4*ir;
                device const ushort * q2 = q1+32;
                scales[0]=sc[0]&0x3f3f; scales[1]=sc[2]&0x3f3f;
                scales[2]=(sc[4]&0x0f0f)|((sc[0]&0xc0c0)>>2);
                scales[3]=((sc[4]>>4)&0x0f0f)|((sc[2]&0xc0c0)>>2);
                float4 v1=0.0f, v2=0.0f;
                FOR_UNROLL (short i=0; i<4; ++i) {
                    v1[0]+=yl[2*i]*(q1[i]&0x000f);
                    v1[1]+=yl[2*i+1]*(q1[i]&0x0f00);
                    v1[2]+=yl[2*i+8]*(q1[i]&0x00f0);
                    v1[3]+=yl[2*i+9]*(q1[i]&0xf000);
                    v2[0]+=yh[2*i]*(q2[i]&0x000f);
                    v2[1]+=yh[2*i+1]*(q2[i]&0x0f00);
                    v2[2]+=yh[2*i+8]*(q2[i]&0x00f0);
                    v2[3]+=yh[2*i+9]*(q2[i]&0xf000);
                }
                sums[p*ROWS+r] += w->d*((v1[0]+v1[1]/256.0f)*s[0] +
                    (v1[2]+v1[3]/256.0f)*s[1]/16.0f +
                    (v2[0]+v2[1]/256.0f)*s[4] + (v2[2]+v2[3]/256.0f)*s[5]/16.0f) -
                    w->dmin*(sy[0]*s[2]+sy[1]*s[3]+sy[2]*s[6]+sy[3]*s[7]);
            }
        }
        y4+=4*256;
    }
    for (short r=0; r<ROWS; ++r) {
        const float gate=simd_sum(sums[r]);
        const float up=simd_sum(sums[ROWS+r]);
        if (lane==0 && row0+r<a.ne01) {
            ((device float *)(output + uint64_t(seq_col*a.ne1+slot)*a.nb1))[row0+r] = (gate/(1.0f+exp(-gate)))*up;
        }
    }
}
typedef decltype(kernel_mul_mv_id_q4_gateup<2>) q4_gateup_t;
template [[host_name("kernel_mul_mv_id_q4_gateup_r1")]] kernel q4_gateup_t kernel_mul_mv_id_q4_gateup<1>;
template [[host_name("kernel_mul_mv_id_q4_gateup_r2")]] kernel q4_gateup_t kernel_mul_mv_id_q4_gateup<2>;
template [[host_name("kernel_mul_mv_id_q4_gateup_r4")]] kernel q4_gateup_t kernel_mul_mv_id_q4_gateup<4>;

template<int nr0, typename args_t>
void kernel_mul_mv_q4_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[16];
    float yh[16];

    float sumf[nr0]={0.f};

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01/2;
            sc += args.nb01/2;
            dh += args.nb01/2;
        }

        y4 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q5_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q5_K * x = (device const block_q5_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0]={0.f};

    float yl[16], yh[16];

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short iq  = tid/4;
    const short ir  = tid%4;

    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;

    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += 4) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const half * dh = &x[i].d;
        device const uint16_t * a = (device const uint16_t *)x[i].scales + iq;

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        for (short row = 0; row < nr0; ++row) {
            device const uint8_t * q2 = q1 + 64;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f};
            float4 acc2 = {0.f};
            FOR_UNROLL (short l = 0; l < 8; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * (q1[l] & 0x0F);
                acc1[1] += yl[l+8] * (q1[l] & 0xF0);
                acc1[2] += yh[l+0] * (q2[l] & 0x0F);
                acc1[3] += yh[l+8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l+0] : 0.f;
                acc2[1] += h & hm2 ? yl[l+8] : 0.f;
                acc2[2] += h & hm3 ? yh[l+0] : 0.f;
                acc2[3] += h & hm4 ? yh[l+8] : 0.f;
            }

            sumf[row] += dh[0] * (sc8[0] * (acc1[0]      + 16.f*acc2[0]) +
                                  sc8[1] * (acc1[1]/16.f + 16.f*acc2[1]) +
                                  sc8[4] * (acc1[2]      + 16.f*acc2[2]) +
                                  sc8[5] * (acc1[3]/16.f + 16.f*acc2[3])) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01;
            qh += args.nb01;
            dh += args.nb01/2;
            a  += args.nb01/2;
        }

        y1 += 4 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q5_K_f32")]]
kernel void kernel_mul_mv_q5_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q6_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0] = { 0.f };

    float yl[16];

    const short tid = tiisg/2;
    const short ix  = tiisg%2;
    const short ip  = tid/8;         // 0 or 1
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;
        device const half    * dh = &x[i].d;

        device const float * y = yy + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01/2;
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q6_K_f32")]]
kernel void kernel_mul_mv_q6_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// ======================= "True" 2-bit

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + aux8[l]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    sum += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += args.nb01/2;
            q2 += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xxs_f32")]]
kernel void kernel_mul_mv_iq2_xxs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xs * x = (device const block_iq2_xs *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 512);
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2xs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const uint8_t  * sc = xr->scales + ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const uint8_t ls1 = sc[0] & 0xf;
            const uint8_t ls2 = sc[0] >>  4;
            const float d1 = db * (0.5f + ls1);
            const float d2 = db * (0.5f + ls2);

            float sum1 = 0, sum2 = 0;
            for (short l = 0; l < 2; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum1 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (short l = 2; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum2 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;

            dh += args.nb01/2;
            q2 += args.nb01/2;
            sc += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xs_f32")]]
kernel void kernel_mul_mv_iq2_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl<N_R0_IQ2_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// FC_mul_mv_split: for nb32 < 32 (nb32 divides 32), 32/nb32 threads share each chunk and each takes a slice of the rows
template<int nr0, typename args_t>
void kernel_mul_mv_iq3_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq3_xxs * x = (device const block_iq3_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * svalues = (threadgroup uint32_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        int nval = 4;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3xxs_grid[pos + i];
        nval = 2;
        pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) ssigns[pos+i] = ksigns_iq2xs[pos+i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const short ntx  = FC_mul_mv_split ? nb32 : 32;
    const short nrep = 32 / ntx;

    const short ix   = tiisg % ntx;
    const short irep = tiisg / ntx;

    const short row0 = (nr0 * irep      ) / nrep;
    const short row1 = (nr0 * (irep + 1)) / nrep;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += ntx) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t  * q3 = xr->qs + 8 * ib + (uint64_t) row0*args.nb01;
        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib + (uint64_t) row0*args.nb01/2;
        device const half * dh = &xr->d + (uint64_t) row0*args.nb01/2;

        for (short row = row0; row < row1; row++) {
            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + q3[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + q3[2*l+1]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += args.nb01/2;
            q3  += args.nb01;
            gas += args.nb01/2;
        }

        y4 += 32 * ntx;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.5f;
        }
    }
}

template<typename args_t>
void kernel_mul_mv_iq3_xxs_f32_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    if (FC_mul_mv_split) {
        kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS_SPLIT, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
    } else {
        kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
    }
}

[[host_name("kernel_mul_mv_iq3_xxs_f32")]]
kernel void kernel_mul_mv_iq3_xxs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_disp<constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq3_s * x = (device const block_iq3_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        int nval = 8;
        int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3s_grid[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 8 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + (ib/2);
        device const uint8_t * signs = xr->signs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? svalues + 256 : svalues;
                const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? svalues + 256 : svalues;
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq3_s_f32")]]
kernel void kernel_mul_mv_iq3_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_s_f32_impl<N_R0_IQ3_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_s * x = (device const block_iq2_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    //threadgroup uint64_t * svalues = (threadgroup uint64_t *) shmem;
    //{
    //    int nval = 32;
    //    int pos  = (32*sgitg + tiisg)*nval;
    //    for (int i = 0; i < nval; ++i) svalues[pos + i] = iq2s_grid[pos + i];
    //    threadgroup_barrier(mem_flags::mem_threadgroup);
    //}

    const short ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 4 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + ib;
        device const uint8_t * signs = qs + QK_K/8;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const float d1 = db * (0.5f + (sc[0] & 0xf));
            const float d2 = db * (0.5f + (sc[0] >>  4));

            float2 sum = {0};
            for (short l = 0; l < 2; ++l) {
                //const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                //const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                for (short j = 0; j < 8; ++j) {
                    sum[0] += yl[8*l + j +  0] * grid1[j] * select(1, -1, signs[l+0] & kmask_iq2xs[j]);
                    sum[1] += yl[8*l + j + 16] * grid2[j] * select(1, -1, signs[l+2] & kmask_iq2xs[j]);
                }
            }
            sumf[row] += d1 * sum[0] + d2 * sum[1];

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_s_f32")]]
kernel void kernel_mul_mv_iq2_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_s_f32_impl<N_R0_IQ2_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq1_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq1_s * x = (device const block_iq1_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    const short ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        float sumy = 0;
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
            sumy += yl[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_s * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint16_t * qh = xr->qh + ib;
        device const half     * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 5) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[0] << 2) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[0] >> 1) & 0x700)));

            float sum = 0;
            for (short j = 0; j < 4; ++j) {
                sum += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                     + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4)
                     + yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                     + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            sumf[row] += (float)dh[0] * (sum + sumy * (qh[0] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA)) * (2*((qh[0] >> 12) & 7) + 1);

            dh += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01/2;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_s_f32")]]
kernel void kernel_mul_mv_iq1_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl<N_R0_IQ1_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq1_m_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq1_m * x = (device const block_iq1_m *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    const short ix = tiisg;

    device const float * y4 = y + 32 * ix;

    iq1m_scale_t scale;

    for (int ib32 = ix; ib32 < nb32; ib32 += 32) {
        float4 sumy = {0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+ 8]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+16]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+24]; sumy[3] += yl[i+24];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_m * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint8_t  * qh = xr->qh + 2 * ib;
        device const uint16_t * sc = (device const uint16_t *)xr->scales;

        for (short row = 0; row < nr0; row++) {
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

            constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
            constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
            constant uint8_t * grid3 = (constant uint8_t *)(iq1s_grid_gpu + (qs[2] | ((qh[1] << 8) & 0x700)));
            constant uint8_t * grid4 = (constant uint8_t *)(iq1s_grid_gpu + (qs[3] | ((qh[1] << 4) & 0x700)));

            float2 sum = {0.f};
            for (short j = 0; j < 4; ++j) {
                sum[0] += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                        + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4);
                sum[1] += yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                        + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);

            sumf[row] += (float)scale.f16 * ((sum[0] + delta1) * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1) +
                                             (sum[1] + delta2) * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1));

            sc += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01;
        }

        y4 += 32 * 32;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_m_f32")]]
kernel void kernel_mul_mv_iq1_m_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl<N_R0_IQ1_M, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int NR0, typename args_t>
void kernel_mul_mv_iq4_nl_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_nl * x = (device const block_iq4_nl *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const int nb   = args.ne00/QK4_NL;
    const int ns01 = args.nb01/args.nb00;

    const short ix = tiisg/2;  // 0...15
    const short it = tiisg%2;  // 0 or 1

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix*QK4_NL + it*8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    // [TAG_MUL_MV_WEIRD]
    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            device const block_iq4_nl & xb = x[row*ns01 + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[2] | (q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 16 * QK4_NL;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq4_nl_f32")]]
kernel void kernel_mul_mv_iq4_nl_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl<N_R0_IQ4_NL, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int NR0, typename args_t>
void kernel_mul_mv_iq4_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_xs * x = (device const block_iq4_xs *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const int nb   = args.ne00/QK_K;
    const int ns01 = args.nb01/args.nb00;

    const short ix = tiisg/16;  // 0 or 1
    const short it = tiisg%16;  // 0...15
    const short ib = it/2;
    const short il = it%2;

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    // [TAG_MUL_MV_WEIRD]
    for (int ibl = ix; ibl < nb && ibl < ns01; ibl += 2) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; ++row) {
            device const block_iq4_xs & xb = x[row*ns01 + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += 2 * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq4_xs_f32")]]
kernel void kernel_mul_mv_iq4_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int NR0, typename args_t>
void kernel_mul_mv_mxfp4_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_mxfp4 * x = (device const block_mxfp4 *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    const int nb   = args.ne00/QK_MXFP4;
    const int ns01 = args.nb01/args.nb00; // this can be larger than nb for permuted src0 tensors

    const short ix = tiisg/2;  // 0...15
    const short it = tiisg%2;  // 0 or 1

    shmem_f32[tiisg] = kvalues_mxfp4_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix*QK_MXFP4 + it*8;

    // note: just the check `ib < nb` is enough, but adding the redundant `&& ib < ns01` check makes the kernel a bit faster
    //       no idea why that is - needs some deeper investigation [TAG_MUL_MV_WEIRD]
    for (int ib = ix; ib < nb && ib < ns01; ib += 16) {
        device const float4 * y4 = (device const float4 *) yb;

        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_mxfp4 & xb = x[row*ns01 + ib];
            device const uint8_t     * q2 = (device const uint8_t *)(xb.qs + 8*it);

            float4 acc1 = yl[0]*float4(shmem_f32[q2[0] &  0x0F], shmem_f32[q2[1] &  0x0F], shmem_f32[q2[2] &  0x0F], shmem_f32[q2[3] &  0x0F]);
            float4 acc2 = yl[1]*float4(shmem_f32[q2[0] >> 4   ], shmem_f32[q2[1] >> 4   ], shmem_f32[q2[2] >> 4   ], shmem_f32[q2[3] >> 4   ]);
            float4 acc3 = yl[2]*float4(shmem_f32[q2[4] &  0x0F], shmem_f32[q2[5] &  0x0F], shmem_f32[q2[6] &  0x0F], shmem_f32[q2[7] &  0x0F]);
            float4 acc4 = yl[3]*float4(shmem_f32[q2[4] >> 4   ], shmem_f32[q2[5] >> 4   ], shmem_f32[q2[6] >> 4   ], shmem_f32[q2[7] >> 4   ]);

            acc1 = (acc1 + acc3) + (acc2 + acc4);

            sumf[row] += e8m0_to_fp32(xb.e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
        }

        yb += 16 * QK_MXFP4;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_mxfp4_f32")]]
kernel void kernel_mul_mv_mxfp4_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_tq2_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_tq2_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_tq2_0 *) ((device char *) src0 + offset0);
    }

    float sumf[nr0] = {0.f};

    // 8 threads per block, NBLOCK blocks per pass, 2 halves per block per pass
    constexpr short NBLOCK = 4;

    constexpr short NB = N_SIMDWIDTH/NBLOCK; // threads per block

    const short blk = tiisg / NB;    // 0..NBLOCK-1, block handled by this thread
    const short htg = tiisg % NB;    // 0..NB-1, thread within block (0..7)

    // byte and y base offsets within the block (32 elements per thread, 4 per byte)
    device const float4 * yb4 = (device const float4 *)(y + 4*htg + blk*QK_K);

    // hoisted per-byte coefficients (from y) and total y-sum, shared across rows
    // ref: https://github.com/ggml-org/llama.cpp/pull/26980
    float4 coef[4];

    for (int ib = blk; ib < nb; ib += NBLOCK) {
        FOR_UNROLL (short h0 = 0; h0 < 2; ++h0) {
            const float4 y0 = yb4[ 0 + 32*h0];
            const float4 y1 = yb4[ 8 + 32*h0];
            const float4 y2 = yb4[16 + 32*h0];
            const float4 y3 = yb4[24 + 32*h0];

            float sumy = 0.f;
            FOR_UNROLL (short j = 0; j < 4; ++j) {
                coef[j] = float4(
                        y0[j],
                        y1[j] - 4.0f*y0[j],
                        y2[j] - 4.0f*y1[j],
                        y3[j] - 4.0f*y2[j]);

                sumy += (y0[j] + y1[j]) + (y2[j] + y3[j]);
            }

            FOR_UNROLL (short row = 0; row < nr0; ++row) {
                device const block_tq2_0 & xb = ax[row][ib];
                device const uchar * qs = xb.qs + 4*htg + 32*h0;

                float sum = -sumy;
                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    // express the 2-bit field shifts (v>>2, v>>4, v>>6) as float floor ops
                    const float v = (float)qs[j];

                    const float f0 = v;
                    const float f1 = floor(v*0.25f);    // v>>2
                    const float f2 = floor(v*0.0625);   // v>>4
                    const float f3 = floor(v*0.015625); // v>>6

                    sum += coef[j][0]*f0 + coef[j][1]*f1 + coef[j][2]*f2 + coef[j][3]*f3;
                }

                sumf[row] += xb.d * sum;
            }
        }

        yb4 += QK_K * NBLOCK / 4;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_tq2_0_f32")]]
kernel void kernel_mul_mv_tq2_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_tq2_0_f32_impl<N_R0_TQ2_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

//
// matrix-vector multiplication
//

typedef void (kernel_mul_mv_disp_t)(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg);

typedef void (kernel_mul_mv2_disp_t)(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg);

template<kernel_mul_mv_disp_t disp_fn>
void mmv_fn(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    disp_fn(args, src0, src1, dst, tgpig, tiisg);
}

template<kernel_mul_mv2_disp_t disp_fn>
void mmv_fn(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    disp_fn(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(mmv_fn<kernel_mul_mv_t_t_disp<half, half, ggml_metal_kargs_mul_mv>>) mul_mv_disp_fn_t;

template<mul_mv_disp_fn_t disp_fn>
kernel void kernel_mul_mv_id(
        constant ggml_metal_kargs_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        device const int * order,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (FC_mul_mv_id_tile_order) {
        const uint row_tile=tgpig.z;
        tgpig.z=tgpig.x;
        tgpig.x=row_tile;
    }
    if (FC_mul_mv_expert_sort) {
        if (FC_mul_mv_expert_sort_tile > 0) {
            const int nr=(args.ne01+args.nr0*FC_mul_mv_nsg-1)/(args.nr0*FC_mul_mv_nsg);
            const int nc=args.nei0*args.nei1;
            const int flat=int(tgpig.x)+int(tgpig.z)*nr;
            const int tile=FC_mul_mv_expert_sort_tile;
            const int rb=(flat/(tile*nc))*tile;
            const int rows_here=min(tile,nr-rb);
            const int within=flat%(tile*nc);
            tgpig.x=rb+within%rows_here;
            tgpig.z=within/rows_here;
        }
        tgpig.z=order[tgpig.z];
    }
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device const char * src0_cur = src0s + i02*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;

    device char * dst_cur = dst + (i1*args.ne0 + i2*args.ne1*args.ne0)*sizeof(float);

    ggml_metal_kargs_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1, // args.ne02,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02, // args.ne02 == 1
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1, // args.ne11,
        /*.ne12 =*/ 1, // args.ne12,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12, // ne12 == 1
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1, // args.ne1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    disp_fn(
        args0,
        /* src0 */ src0_cur,
        /* src1 */ src1_cur,
        /* dst  */ dst_cur,
        shmem,
        tgpig,
        tiitg,
        tiisg,
        sgitg);
}

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_disp<float, float>>>) kernel_mul_mv_id_t;

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_disp<float, float4, float, float4>>>) kernel_mul_mv_id_4_t;

template [[host_name("kernel_mul_mv_id_f32_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_disp<float, float>>>;
template [[host_name("kernel_mul_mv_id_f16_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_disp<half,  float>>>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_id_bf16_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_disp<bfloat, float>>>;
#endif
template [[host_name("kernel_mul_mv_id_f32_f32_4")]]   kernel kernel_mul_mv_id_4_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_disp<float, float4, float, float4>>>;
template [[host_name("kernel_mul_mv_id_f16_f32_4")]]   kernel kernel_mul_mv_id_4_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_disp<half,  half4,  float, float4>>>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_id_bf16_f32_4")]]  kernel kernel_mul_mv_id_4_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_disp<bfloat, bfloat4, float, float4>>>;
#endif

template [[host_name("kernel_mul_mv_id_q8_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>>>;

template [[host_name("kernel_mul_mv_id_q1_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q1_0_f32_impl<N_R0_Q1_0>>>;
template [[host_name("kernel_mul_mv_id_q2_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_0_f32_impl<N_R0_Q2_0>>>;
template [[host_name("kernel_mul_mv_id_q4_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0>>>;
template [[host_name("kernel_mul_mv_id_q4_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_1, N_R0_Q4_1>>>;
template [[host_name("kernel_mul_mv_id_q5_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0>>>;
template [[host_name("kernel_mul_mv_id_q5_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1>>>;

template [[host_name("kernel_mul_mv_id_mxfp4_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>>>;

template [[host_name("kernel_mul_mv_id_q2_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl   <N_R0_Q2_K>>>;
template [[host_name("kernel_mul_mv_id_q3_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q3_K_f32_impl   <N_R0_Q3_K>>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl   <N_R0_Q4_K>>>;
template [[host_name("kernel_mul_mv_id_q5_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_f32_impl   <N_R0_Q5_K>>>;
template [[host_name("kernel_mul_mv_id_q6_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_f32_impl   <N_R0_Q6_K>>>;
template [[host_name("kernel_mul_mv_id_iq1_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_s_f32_impl  <N_R0_IQ1_S>>>;
template [[host_name("kernel_mul_mv_id_iq1_m_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_m_f32_impl  <N_R0_IQ1_M>>>;
template [[host_name("kernel_mul_mv_id_iq2_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS>>>;
template [[host_name("kernel_mul_mv_id_iq2_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xs_f32_impl <N_R0_IQ2_XS>>>;
template [[host_name("kernel_mul_mv_id_iq3_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_xxs_f32_disp<ggml_metal_kargs_mul_mv>>>;
template [[host_name("kernel_mul_mv_id_iq3_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_s_f32_impl  <N_R0_IQ3_S>>>;
template [[host_name("kernel_mul_mv_id_iq2_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_s_f32_impl  <N_R0_IQ2_S>>>;
template [[host_name("kernel_mul_mv_id_iq4_nl_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_nl_f32_impl <N_R0_IQ4_NL>>>;
template [[host_name("kernel_mul_mv_id_iq4_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_xs_f32_impl <N_R0_IQ4_XS>>>;
template [[host_name("kernel_mul_mv_id_tq2_0_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_tq2_0_f32_impl  <N_R0_TQ2_0>>>;

// Q8 small batches with K divisible by 32, including the 320-wide HC projection.
template<short r1ptg, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &) >
void kernel_mul_mv_ext_q4_f32_tail_impl(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 1; // chunks per thread

  //const short nxpsg = (32);
    const short nypsg = (32/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%FC_mul_mv_ne12;
    const int i13 = i1m/FC_mul_mv_ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb; // current chunk index

    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    // reduce only the threads in each row
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }

        //sumf[ir1] = simd_sum(sumf[ir1]);
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

template<short R>
kernel void kernel_mul_mv_ext_q8_tail(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0, device const char * src1, device char * dst,
        uint3 tg [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4_f32_tail_impl<R, block_q8_0, 8, dequantize_q8_0_t4>(args, src0, src1, dst, tg, lane, sg);
}
typedef decltype(kernel_mul_mv_ext_q8_tail<2>) mul_mv_ext_q8_tail_t;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_2_tail")]] kernel mul_mv_ext_q8_tail_t kernel_mul_mv_ext_q8_tail<2>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_3_tail")]] kernel mul_mv_ext_q8_tail_t kernel_mul_mv_ext_q8_tail<3>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_4_tail")]] kernel mul_mv_ext_q8_tail_t kernel_mul_mv_ext_q8_tail<4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_5_tail")]] kernel mul_mv_ext_q8_tail_t kernel_mul_mv_ext_q8_tail<5>;

template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_6")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<6, block_q8_0,   32, dequantize_q8_0_t4>;

template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_7")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<7, block_q8_0,   32, dequantize_q8_0_t4>;

template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_8")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<8, block_q8_0,   32, dequantize_q8_0_t4>;

template [[host_name("kernel_mul_mv_id_q4_K_f32_r1")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl<1>>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32_r4")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl<4>>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32_r8")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl<8>>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32_r16")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl<16>>>;
template [[host_name("kernel_mul_mv_id_q5_1_f32_r2")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_1, 2>>>;
template [[host_name("kernel_mul_mv_id_q5_1_f32_r8")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_1, 8>>>;
template [[host_name("kernel_mul_mv_id_q5_1_f32_r16")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_1, 16>>>;
template<short R>
kernel void kernel_mul_mv_ext_q4_tail(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0, device const char * src1, device char * dst,
        uint3 tg [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ext_q4_f32_tail_impl<R, block_q4_0, 8, dequantize_q4_0_t4>(args, src0, src1, dst, tg, lane, sg);
}
typedef decltype(kernel_mul_mv_ext_q4_tail<2>) mul_mv_ext_q4_tail_t;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_2_tail")]] kernel mul_mv_ext_q4_tail_t kernel_mul_mv_ext_q4_tail<2>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_3_tail")]] kernel mul_mv_ext_q4_tail_t kernel_mul_mv_ext_q4_tail<3>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_4_tail")]] kernel mul_mv_ext_q4_tail_t kernel_mul_mv_ext_q4_tail<4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_5_tail")]] kernel mul_mv_ext_q4_tail_t kernel_mul_mv_ext_q4_tail<5>;


// Bounded expert groups reuse packed weights with eight values per lane.
// Map construction runs once per operator; each output occurrence has one owner.
template<short C>
kernel void kernel_mul_mv_id_group_map(
        constant ggml_metal_kargs_mul_mv_id & a, device const char * ids,
        device int * map, uint p [[thread_position_in_grid]]) {
    const uint count=uint(a.nei0*a.nei1);
    if (p>=count) return;
    const int expert=((device const int *)(ids+(p/a.nei0)*a.nbi1))[p%a.nei0];
    int before=0;
    for(uint j=0;j<p;++j) before+=((device const int *)(ids+(j/a.nei0)*a.nbi1))[j%a.nei0]==expert;
    FOR_UNROLL(short c=0;c<C;++c) map[p*C+c]=-1;
    if(before%C) { map[p*C]=-2; return; }
    map[p*C]=int(p);
    short c=1;
    for(uint j=p+1;j<count && c<C;++j) {
        if(((device const int *)(ids+(j/a.nei0)*a.nbi1))[j%a.nei0]==expert) map[p*C+c++]=int(j);
    }
}
typedef decltype(kernel_mul_mv_id_group_map<1>) expert_group_map_t;
template [[host_name("kernel_mul_mv_id_group_map1")]] kernel expert_group_map_t kernel_mul_mv_id_group_map<1>;
template [[host_name("kernel_mul_mv_id_group_map2")]] kernel expert_group_map_t kernel_mul_mv_id_group_map<2>;
template [[host_name("kernel_mul_mv_id_group_map4")]] kernel expert_group_map_t kernel_mul_mv_id_group_map<4>;

inline void expert_unpack8(device const block_q4_K * w, int q,
        thread float4 & lo, thread float4 & hi, thread float & d, thread float & m) {
    const int s=q/32;
    const uint scale=s<4 ? (w->scales[s]&63u) : ((w->scales[s+4]&15u)|((w->scales[s-4]>>6)<<4));
    const uint minimum=s<4 ? (w->scales[s+4]&63u) : ((w->scales[s+4]>>4)|((w->scales[s]>>6)<<4));
    const uint2 packed=*((device const uint2 *)(w->qs+(s/2)*32+(q%32)));
    const uint shift=(s&1)*4;
    FOR_UNROLL(short j=0;j<4;++j) {
        lo[j]=float((packed[0]>>(8*j+shift))&15u);
        hi[j]=float((packed[1]>>(8*j+shift))&15u);
    }
    d=float(w->d)*float(scale); m=-float(w->dmin)*float(minimum);
}
inline void expert_unpack8(device const block_q5_1 * w, int q,
        thread float4 & lo, thread float4 & hi, thread float & d, thread float & m) {
    const uint2 packed=*((device const uint2 *)(w->qs+(q%16)));
    const uint high=*((device const uint *)w->qh);
    const uint shift=q>=16?4:0;
    FOR_UNROLL(short j=0;j<4;++j) {
        lo[j]=float(((packed[0]>>(8*j+shift))&15u)|(((high>>(q+j))&1u)<<4));
        hi[j]=float(((packed[1]>>(8*j+shift))&15u)|(((high>>(q+j+4))&1u)<<4));
    }
    d=float(w->d);m=float(w->m);
}

template<typename QT,short QK,short NX,short ROWS,short C>
kernel void kernel_mul_mv_id_group8(
        constant ggml_metal_kargs_mul_mv_id & a,
        device const char * weights, device const char * input, device char * output,
        device const char * ids, device const int * map,
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const int flat=group.z;
    if(map[flat*C]<0) return;
    const int expert=((device const int *)(ids+(flat/a.nei0)*a.nbi1))[flat%a.nei0];
    constexpr short NY=32/NX;
    const int row0=((group.x*FC_mul_mv_nsg+sg)*NY+lane/NX)*ROWS;
    const int tx=lane%NX;
    int owner[C];
    device const float4 * x[C];
    FOR_UNROLL(short c=0;c<C;++c) {
        owner[c]=map[flat*C+c];
        const int p=max(owner[c],0);
        x[c]=(device const float4 *)(input+(p/a.nei0)*a.nb12+((p%a.nei0)%a.ne11)*a.nb11);
    }
    float sum[ROWS][C]={};
    for(int chunk=tx;chunk<a.ne00/8;chunk+=NX) {
        float4 v0[C],v1[C];float vs[C];
        FOR_UNROLL(short c=0;c<C;++c) if(owner[c]>=0) {
            v0[c]=x[c][2*chunk];v1[c]=x[c][2*chunk+1];
            vs[c]=(v0[c][0]+v0[c][1]+v0[c][2]+v0[c][3])+(v1[c][0]+v1[c][1]+v1[c][2]+v1[c][3]);
        }
        FOR_UNROLL(short r=0;r<ROWS;++r) {
            device const QT * w=(device const QT *)(weights+expert*a.nb02+min(row0+r,a.ne01-1)*a.nb01)+chunk/(QK/8);
            float4 q0,q1;float d,m;
            expert_unpack8(w,(chunk%(QK/8))*8,q0,q1,d,m);
            FOR_UNROLL(short c=0;c<C;++c) if(owner[c]>=0) {
                sum[r][c]+=d*(dot(q0,v0[c])+dot(q1,v1[c]))+m*vs[c];
            }
        }
    }
    FOR_UNROLL(short r=0;r<ROWS;++r) {
        FOR_UNROLL(short c=0;c<C;++c) if(owner[c]>=0) {
            float s=sum[r][c];
            FOR_UNROLL(short off=NX/2;off>0;off/=2) s+=simd_shuffle_down(s,off);
            if(tx==0 && row0+r<a.ne01) {
                const int p=owner[c];
                ((device float *)(output+((p/a.nei0)*a.ne1+p%a.nei0)*a.nb1))[row0+r]=s;
            }
        }
    }
}
typedef decltype(kernel_mul_mv_id_group8<block_q4_K,256,32,1,1>) expert_group8_t;
#define EXP_GROUP(Q,K,X,R,C) template [[host_name("kernel_mul_mv_id_" #Q "_f32_g" #C "x" #X "r" #R)]] kernel expert_group8_t kernel_mul_mv_id_group8<block_##Q,K,X,R,C>;
#define EXP_GROUP_COLS(Q,K,X,R) EXP_GROUP(Q,K,X,R,1) EXP_GROUP(Q,K,X,R,2) EXP_GROUP(Q,K,X,R,4)
#define EXP_GROUP_ROWS(Q,K,X) EXP_GROUP_COLS(Q,K,X,1) EXP_GROUP_COLS(Q,K,X,2)
EXP_GROUP_ROWS(q4_K,256,8) EXP_GROUP_ROWS(q4_K,256,16) EXP_GROUP_ROWS(q4_K,256,32)
EXP_GROUP_ROWS(q5_1,32,8) EXP_GROUP_ROWS(q5_1,32,16) EXP_GROUP_ROWS(q5_1,32,32)
#undef EXP_GROUP_ROWS
#undef EXP_GROUP_COLS
#undef EXP_GROUP

// Stable rank permutation changes dispatch order, preserving every output slot.
kernel void kernel_mul_mv_id_sort(
        constant ggml_metal_kargs_mul_mv_id & a, device const char * ids,
        device int * order, uint p [[thread_position_in_grid]]) {
    const uint count=uint(a.nei0*a.nei1);
    if(p>=count) return;
    const int expert=((device const int *)(ids+(p/a.nei0)*a.nbi1))[p%a.nei0];
    uint rank=0;
    for(uint j=0;j<count;++j) {
        const int other=((device const int *)(ids+(j/a.nei0)*a.nbi1))[j%a.nei0];
        rank+=other<expert || (other==expert && j<p);
    }
    order[rank]=int(p);
}

// Direct selected-row projection avoids a full F32 gather intermediate.
template<typename QT>
kernel void kernel_mul_mv_selected(
        constant uint64_t * a, device const char * weights,
        device const float * x, device const int * ids, device float * output,
        uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr short ROWS=2;
    const int k=a[0],n=a[1],vocab=a[3];
    const int first=(group.x*2+sg)*ROWS;
    const int bi=lane/4,qi=(lane%4)*8;
    device const QT * w[ROWS];
    FOR_UNROLL(short r=0;r<ROWS;++r) {
        const int id=ids[min(first+r,n-1)];
        w[r]=(device const QT *)(weights+uint64_t(clamp(id,0,vocab-1))*a[2]);
    }
    float sums[ROWS]={};
    for(int b=bi;b<k/32;b+=8) {
        float values[8];
        FOR_UNROLL(short j=0;j<8;++j) values[j]=x[b*32+qi+j];
        FOR_UNROLL(short r=0;r<ROWS;++r) {
            float dotq=0.0f;
            FOR_UNROLL(short j=0;j<8;++j) {
                float q;
                if(is_same<QT,block_q4_0>::value) {
                    const int p=qi+j;
                    q=float(int((w[r][b].qs[p%16]>>(p>=16?4:0))&15)-8);
                } else q=float(w[r][b].qs[qi+j]);
                dotq+=q*values[j];
            }
            sums[r]+=float(w[r][b].d)*dotq;
        }
    }
    FOR_UNROLL(short r=0;r<ROWS;++r) {
        const float value=simd_sum(sums[r]);
        if(lane==0 && first+r<n) output[first+r]=value;
    }
}
typedef decltype(kernel_mul_mv_selected<block_q4_0>) selected_projection_t;
template [[host_name("kernel_mul_mv_selected_q4_0")]] kernel selected_projection_t kernel_mul_mv_selected<block_q4_0>;
template [[host_name("kernel_mul_mv_selected_q8_0")]] kernel selected_projection_t kernel_mul_mv_selected<block_q8_0>;

// Single-token Q4 head: packed eight-byte reads and shared activations across rows.
// The four partial sums retain the existing Q4 affine block evaluation order.
template<short ROWS>
kernel void kernel_mul_mv_r95_head_q4(constant ulong * p,
        device const char * weights,device const float * y,device float * output,
        uint group [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint k=uint(p[0]),m=uint(p[1]);
    const uint r0=(group*4+sg)*ROWS;
    const uint il=(lane%2)*8;
    float sums[ROWS]={};
    for(uint b=lane/2;b<k/32;b+=16) {
        const float4 lo0=*((device const float4 *)(y+b*32+il));
        const float4 lo1=*((device const float4 *)(y+b*32+il+4));
        const float4 hi0=*((device const float4 *)(y+b*32+il+16));
        const float4 hi1=*((device const float4 *)(y+b*32+il+20));
        float a[16]={lo0.x,lo0.y/256.0f,lo0.z,lo0.w/256.0f,
                     lo1.x,lo1.y/256.0f,lo1.z,lo1.w/256.0f,
                     hi0.x/16.0f,hi0.y/4096.0f,hi0.z/16.0f,hi0.w/4096.0f,
                     hi1.x/16.0f,hi1.y/4096.0f,hi1.z/16.0f,hi1.w/4096.0f};
        const float sumy=((lo0.x+lo0.y)+(lo0.z+lo0.w))+((lo1.x+lo1.y)+(lo1.z+lo1.w))+
                         ((hi0.x+hi0.y)+(hi0.z+hi0.w))+((hi1.x+hi1.y)+(hi1.z+hi1.w));
        FOR_UNROLL(short r=0;r<ROWS;++r) {
            device const block_q4_0 * w=(device const block_q4_0 *)(weights+ulong(min(r0+r,m-1))*p[2])+b;
            const ushort4 q=ushort4(*((device const packed_ushort4 *)(w->qs+il)));
            float v0=0,v1=0,v2=0,v3=0;
            FOR_UNROLL(short j=0;j<4;++j) {
                v0+=a[2*j]*(q[j]&0x000f); v1+=a[2*j+1]*(q[j]&0x0f00);
                v2+=a[2*j+8]*(q[j]&0x00f0); v3+=a[2*j+9]*(q[j]&0xf000);
            }
            sums[r]+=float(w->d)*((((sumy*-8.0f+v0)+v1)+v2)+v3);
        }
    }
    FOR_UNROLL(short r=0;r<ROWS;++r) {const float v=simd_sum(sums[r]);if(lane==0 && r0+r<m)output[r0+r]=v;}
}
typedef decltype(kernel_mul_mv_r95_head_q4<2>) r95_head_t;
template [[host_name("kernel_mul_mv_r95_head_q4_r2")]] kernel r95_head_t kernel_mul_mv_r95_head_q4<2>;
template [[host_name("kernel_mul_mv_r95_head_q4_r4")]] kernel r95_head_t kernel_mul_mv_r95_head_q4<4>;
template [[host_name("kernel_mul_mv_r95_head_q4_r8")]] kernel r95_head_t kernel_mul_mv_r95_head_q4<8>;
