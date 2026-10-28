// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
//
// A C-flavoured, driver-shaped header: the input half of this backend's tests.
// Every construct here exists to produce something specific in expected/, so
// the pair reads as "exactly what goes in, exactly what comes out".
//
// The shape is a *host* C API, as opposed to the C++ device headers the Numba
// backends were built for. Nothing here is __device__, which is why this
// backend defaults Skip Non Device to False.

// Status enum, named by a tag the public API never uses.
typedef enum AcmeResult_enum {
  ACME_SUCCESS = 0,
  ACME_ERROR_NOT_FOUND = 500
} AcmeResult;

// Opaque handles: pointers to structs that are never defined.
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

// An anonymous enum: no tag, so the typedef name is the only name it has. A
// name not recognised as an enum lowers to the opaque pointer type, which is
// wrong by ABI rather than merely by spelling.
typedef enum {
  ACME_RESOURCE_TYPE_INVALID = 0,
  ACME_RESOURCE_TYPE_SM = 1
} AcmeResourceType;

extern AcmeResult acmeGetResource(AcmeStream hStream, AcmeResourceType type);

// A descriptor reached only through a field. The dialect has to declare every
// record reachable by field access, not just the ones a signature names, so
// AcmeExtent must appear in the type vocabulary even though no op mentions it.
typedef struct AcmeExtent_st {
  unsigned int width;
  unsigned int height;
} AcmeExtent;

typedef struct AcmeDesc_st {
  AcmeExtent extent;
  // A callback field is an opaque pointer, not a record named after its
  // signature.
  void (*onComplete)(void *userData);
} AcmeDesc;

extern AcmeResult acmeStreamConfigure(AcmeStream hStream,
                                      const AcmeDesc *desc);

// Two unnamed structs under one parent, each reaching a *different* further
// record. Nothing in the AST distinguishes siblings but their order, so if the
// resolver cannot count fields both arms collapse onto one and the other's
// record goes missing entirely -- a missing type rather than a duplicated name.
typedef struct AcmeLocation_st {
  int deviceId;
} AcmeLocation;

typedef struct AcmeOffset_st {
  unsigned int bytes;
} AcmeOffset;

typedef struct AcmeOperand_st {
  struct {
    AcmeLocation loc;
  } ptr;
  struct {
    AcmeOffset off;
  } array;
} AcmeOperand;

extern AcmeResult acmeStreamOperate(AcmeStream hStream,
                                    const AcmeOperand *operand);

// An anonymous *field* holding an anonymous record. There is no member name to
// name the record after, so it falls back to the field's index: without that
// the name degenerates to <Parent>_ and camel-cases into a case-variant of the
// parent, which TableGen accepts and the dialect then rejects at registration.
typedef struct AcmeDevResource_st {
  int type;
  char _padding[8];
  union {
    unsigned int smCount;
  };
} AcmeDevResource;

extern AcmeResult acmeDeviceGetResource(const AcmeDevResource *resource);
