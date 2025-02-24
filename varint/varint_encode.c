#include <stdint.h>
#include <arm_neon.h>

static inline uint32_t arm64_clz(uint32_t x) {
    return __builtin_clz(x);
}

__attribute__((always_inline)) 
static inline uint8x8_t gen_msb_mask(uint32_t count) {
    const uint64_t masks[] = {
        0x0000000000000080, // count=1: [0]
        0x0000000000808080, // count=2: [0][1]
        0x0000808080808080, // count=3: [0][1][2]
        0x8080808080808080  // count=4: [0][1][2][3]
    };
    count = count < 1 ? 1 : (count > 4 ? 4 : count);
    return vcreate_u8(masks[count-1]);
}

char* EncodeVarint(char* __restrict dst, uint32_t v) {
    // 阶段1：精确计算长度
    const uint32_t lz = __builtin_clz(v | 1);
    uint32_t count = (32 - lz + 6) / 7; // 标准公式
    
    // 阶段2：正确的位域分割
    const int32_t shifts[] = {-0, -7, -14, -21};
    int32x4_t shift_vec = vld1q_s32(shifts);
    uint32x4_t vec = vshlq_u32(vdupq_n_u32(v), shift_vec);
    vec = vandq_u32(vec, vdupq_n_u32(0x7F));

    // 阶段3：小端序转换
    uint8x8_t seg_bytes = vrev32_u8(vmovn_u16(
        vcombine_u16(vmovn_u32(vec), vdup_n_u16(0))
    ));
    
    // 阶段4：设置MSB
    uint8x8_t packed = vorr_u8(
        seg_bytes,
        gen_msb_mask(count)
    );

    // 阶段5：正确存储
    const uint8x8_t store_mask = vld1_u8(
        (const uint8_t[]){0xFF,0xFF,0xFF,0xFF,0,0,0,0} + 4 - count
    );
    
    uint8x8_t dst_vec = vld1_u8((uint8_t*)dst);
    dst_vec = vbsl_u8(store_mask, packed, dst_vec);
    vst1_u8((uint8_t*)dst, dst_vec);

    return dst + count;
}