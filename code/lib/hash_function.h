#ifndef MURMUR128_SPLIT_C11_HPP
#define MURMUR128_SPLIT_C11_HPP

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <utility>

// MurmurHash3 x64 128-bit
// Trả về pair {h1,h2}
inline std::pair<uint64_t,uint64_t>
MurmurHash3_x64_128(const void* key, size_t len, uint64_t seed) {
    const uint8_t* data = (const uint8_t*)key;
    const size_t nblocks = len / 16;

    uint64_t h1 = seed;
    uint64_t h2 = seed;

    const uint64_t c1 = 0x87c37b91114253d5ULL;
    const uint64_t c2 = 0x4cf5ad432745937fULL;

    // body
    const uint64_t* blocks = (const uint64_t*)(data);
    for (size_t i = 0; i < nblocks; ++i) {
        uint64_t k1 = blocks[i*2+0];
        uint64_t k2 = blocks[i*2+1];

        k1 *= c1; k1 = (k1 << 31) | (k1 >> (64-31)); k1 *= c2; h1 ^= k1;
        h1 = (h1 << 27) | (h1 >> (64-27)); h1 += h2; h1 = h1*5 + 0x52dce729;

        k2 *= c2; k2 = (k2 << 33) | (k2 >> (64-33)); k2 *= c1; h2 ^= k2;
        h2 = (h2 << 31) | (h2 >> (64-31)); h2 += h1; h2 = h2*5 + 0x38495ab5;
    }

    // tail
    const uint8_t* tail = (const uint8_t*)(data + nblocks*16);
    uint64_t k1 = 0;
    uint64_t k2 = 0;

    switch (len & 15) {
    case 15: k2 ^= ((uint64_t)tail[14]) << 48;
    case 14: k2 ^= ((uint64_t)tail[13]) << 40;
    case 13: k2 ^= ((uint64_t)tail[12]) << 32;
    case 12: k2 ^= ((uint64_t)tail[11]) << 24;
    case 11: k2 ^= ((uint64_t)tail[10]) << 16;
    case 10: k2 ^= ((uint64_t)tail[9]) << 8;
    case 9:  k2 ^= ((uint64_t)tail[8]) << 0;
             k2 *= c2; k2 = (k2 << 33) | (k2 >> (64-33)); k2 *= c1; h2 ^= k2;
    case 8:  k1 ^= ((uint64_t)tail[7]) << 56;
    case 7:  k1 ^= ((uint64_t)tail[6]) << 48;
    case 6:  k1 ^= ((uint64_t)tail[5]) << 40;
    case 5:  k1 ^= ((uint64_t)tail[4]) << 32;
    case 4:  k1 ^= ((uint64_t)tail[3]) << 24;
    case 3:  k1 ^= ((uint64_t)tail[2]) << 16;
    case 2:  k1 ^= ((uint64_t)tail[1]) << 8;
    case 1:  k1 ^= ((uint64_t)tail[0]) << 0;
             k1 *= c1; k1 = (k1 << 31) | (k1 >> (64-31)); k1 *= c2; h1 ^= k1;
    };

    // finalization
    h1 ^= (uint64_t)len;
    h2 ^= (uint64_t)len;

    h1 += h2;
    h2 += h1;

    // fmix64
    const uint64_t cfm1 = 0xff51afd7ed558ccdULL;
    const uint64_t cfm2 = 0xc4ceb9fe1a85ec53ULL;

    uint64_t fmix64 = h1;
    fmix64 ^= fmix64 >> 33;
    fmix64 *= cfm1;
    fmix64 ^= fmix64 >> 33;
    fmix64 *= cfm2;
    fmix64 ^= fmix64 >> 33;
    h1 = fmix64;

    fmix64 = h2;
    fmix64 ^= fmix64 >> 33;
    fmix64 *= cfm1;
    fmix64 ^= fmix64 >> 33;
    fmix64 *= cfm2;
    fmix64 ^= fmix64 >> 33;
    h2 = fmix64;

    h1 += h2;
    h2 += h1;

    return std::make_pair(h1,h2);
}

// hash cho u64
inline std::pair<uint64_t,uint64_t>
MurmurHash3_x64_128_u64(uint64_t key, uint64_t seed) {
    uint64_t buf[2] = { key, 0 };
    return MurmurHash3_x64_128(buf, sizeof(buf), seed);
}

// reduce unbiased: map hash -> [0,buckets)
inline uint64_t reduce_u64_to_range(uint64_t h, uint64_t buckets) {
    __uint128_t prod = ( __uint128_t)h * ( __uint128_t)buckets;
    return (uint64_t)(prod >> 64);
}

// Lấy 2 bucket từ key u64
inline std::pair<uint64_t,uint64_t>
buckets_from_u64(uint64_t key, uint64_t seed, uint64_t buckets) {
    std::pair<uint64_t,uint64_t> h = MurmurHash3_x64_128_u64(key, seed);
    uint64_t b1 = reduce_u64_to_range(h.first, buckets);
    uint64_t b2 = reduce_u64_to_range(h.second, buckets);
    return std::make_pair(b1,b2);
}

#endif