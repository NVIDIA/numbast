// Without `cuda_wrappers/` include overrides in the clang++ include paths,
// the below include will raise a error for the __noinline_ macro somewhere
// in the call stack. See the error being discussed here:
// https://github.com/NVIDIA/thrust/issues/1703
// The fix is included in clang17+:
// https://github.com/llvm/llvm-project-release-prs/pull/698/files
#include <cooperative_groups/reduce.h>
