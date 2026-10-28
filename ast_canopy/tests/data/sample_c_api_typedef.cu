// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on
//
// The typedef forms a C library uses to declare its public type vocabulary.
// None of these alias a record directly, which is why they were previously
// dropped.

// An opaque handle. The underlying type is a *pointer* to an incomplete
// struct, so the record is one indirection away.
typedef struct AcmeStream_st *AcmeStream;

// A status type. The underlying type is a tag, but not a record.
typedef enum AcmeResult_enum {
  ACME_SUCCESS = 0,
  ACME_ERROR_NOT_FOUND = 500
} AcmeResult;

// An anonymous enum. There is no tag, so the typedef name is the only name
// this type will ever have.
typedef enum {
  ACME_RESOURCE_TYPE_INVALID = 0,
  ACME_RESOURCE_TYPE_SM = 1
} AcmeResourceType;

// A scalar alias, carrying no record anywhere.
typedef unsigned long long AcmeDeviceptr;
