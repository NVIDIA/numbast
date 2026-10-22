// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
//
// Example header meant to express the features of a CUDA driver shaped C library.


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

// a parameter name that collides with a generated operation accessor
extern AcmeResult acmeStreamSetLimit(AcmeStream hStream, unsigned long value);

// a by-value aggregate that needs a modelled dialect type
typedef struct AcmeHandle_st {
  char reserved[64];
} AcmeHandle;

extern AcmeResult acmeStreamOpenHandle(AcmeStream *phStream, AcmeHandle handle);

// a void function has no result
extern void acmeShutdown(void);

// anonymous enum typedef: ast_canopy does not report it, and there is no tag
// to recover it from, so it has to be found by its public name or it maps to
// the opaque pointer type instead of an integer
typedef enum {
  ACME_RESOURCE_TYPE_INVALID = 0,
  ACME_RESOURCE_TYPE_SM = 1
} AcmeResourceType;

extern AcmeResult acmeGetResource(AcmeStream hStream, AcmeResourceType type);
