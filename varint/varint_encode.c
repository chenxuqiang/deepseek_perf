#include <stdint.h>
#include <arm_neon.h>

static inline uint32_t arm64_clz(uint32_t x) {
    return __builtin_clz(x);
}

__attribute__((always_inline)) 
static inline uint8x8_t gen_msb_mask(uint32_t count) {
    const uint64_t masks[] = {
        0x0000000000000000, // count=1
        0x0000000000000080, // count=2
        0x0000000000008080, // count=3
        0x0000000000808080  // count=4
    };
    return vcreate_u8(masks[count-1]);
}

char* EncodeVarint(char* __restrict dst, uint32_t v) {
    // 阶段1：修正魔法数计算
    const uint32_t lz = __builtin_clz(v | 1);
    const uint32_t count = (0x1FCF5BDE >> (lz * 2)) & 0x3;
    
    // 阶段2：正确NEON移位（关键修复）
    const int32_t shifts[] = {0, -7, -14, -21}; // 负号表示右移
    int32x4_t shift_vec = vld1q_s32(shifts);
    uint32x4_t vec = vshlq_u32(vdupq_n_u32(v), shift_vec);
    vec = vandq_u32(vec, vdupq_n_u32(0x7F));

    // 阶段3：向量构造优化
    uint8x8_t seg_bytes = vmovn_u16(
        vcombine_u16(vmovn_u32(vec), vdup_n_u16(0))
    );
    
    // 阶段4：并行设置MSB
    uint8x8_t packed = vorr_u8(
        seg_bytes,
        gen_msb_mask(count)
    );

    // 阶段5：SIMD存储优化
    const uint8x8_t store_mask = vld1_u8(
        (const uint8_t[]){0xFF,0xFF,0xFF,0xFF,0,0,0,0} + 4 - count
    );
    
    uint8x8_t dst_vec = vld1_u8((uint8_t*)dst);
    dst_vec = vbsl_u8(store_mask, packed, dst_vec);
    vst1_u8((uint8_t*)dst, dst_vec);

    return dst + count; // 修正返回偏移量
}