template <typename T, int BLOCK_DIM_X, int ITEMS_PER_THREAD> class BlockLoad {
public:
  struct TempStorage {};

  __device__ BlockLoad() {}
  __device__ explicit BlockLoad(TempStorage &) {}

  // Single-item load: just assign input to output
  __device__ void Load(T input, T &output) { output = input; }

  // Array load: copy ITEMS_PER_THREAD elements from input to output
  __device__ void Load(T(input)[ITEMS_PER_THREAD],
                       T (&output)[ITEMS_PER_THREAD]) {
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
      output[i] = input[i];
    }
  }
};
