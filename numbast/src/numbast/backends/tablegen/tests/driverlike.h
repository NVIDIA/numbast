// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
//
// A C-flavoured, driver-shaped header: status-returning entry points, opaque
// handles behind pointer-to-incomplete-struct typedefs, and out-parameters.
// This is the shape the TableGen backend targets: a *host* C API, as opposed
// to the C++ device headers the Numba backends were built for. Nothing here is
// __device__, which is why this backend defaults skip_non_device to False.

typedef enum AcmeResult_enum {
  ACME_SUCCESS = 0,
  ACME_ERROR_NOT_FOUND = 500
} AcmeResult;

typedef struct AcmeStream_st *AcmeStream;
typedef struct AcmeEvent_st *AcmeEvent;

// out-param + scalar input
extern AcmeResult acmeStreamCreate(AcmeStream *phStream, unsigned int Flags);

// all-handle inputs, no out-param
extern AcmeResult acmeStreamWaitEvent(AcmeStream hStream, AcmeEvent hEvent,
                                      unsigned int Flags);

// out-param of scalar type
extern AcmeResult acmeDeviceGetCount(int *count);

// no status: a plain value-returning entry point
extern int acmeVersion(void);

// a parameter named after a C++ keyword, which is legal in the shim but not in
// TableGen -- the two sanitizers must not be shared
extern AcmeResult acmeStreamSetLimit(AcmeStream hStream, unsigned long value);

// by-value aggregate: no header-free spelling, so the shim needs a size-only
// POD
typedef struct AcmeHandle_st {
  char reserved[64];
} AcmeHandle;

extern AcmeResult acmeStreamOpenHandle(AcmeStream *phStream, AcmeHandle handle);

// returns nothing at all: the shim must not emit `return`
extern void acmeShutdown(void);

// anonymous enum typedef: ast_canopy does not report it, and there is no tag
// to recover it from, so it has to be found by its public name or it maps to
// the opaque pointer type instead of an integer
typedef enum {
  ACME_RESOURCE_TYPE_INVALID = 0,
  ACME_RESOURCE_TYPE_SM = 1
} AcmeResourceType;

extern AcmeResult acmeGetResource(AcmeStream hStream, AcmeResourceType type);
