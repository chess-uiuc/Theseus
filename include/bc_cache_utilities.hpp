// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

namespace Theseus
{

  enum class BCType : int
    {
      Invalid = -1,
      SlipWall = 0,
      SupersonicInflow = 1,
      SupersonicOutflow = 2,
      PrescribedState = 3,
      Symmetry = 4,
      Axis = 5,
      NoSlipAdiab = 6,
      NoSlipIso = 7,
      NumBCTypes = 8
    };

  enum class BCDataKind : int
    {
      None = 0,
      ScalarConstant = 1,
      VectorConstant = 2,
      VectorAndScalarConstant = 3,
      NumBCDataKinds = 4
    };

  struct BCDescriptor
  {
    int type;       // BCType
    int data_kind;  // BCDataKind
    int data_index; // offset/index into packed scalar/vector tables
    int flags;      // reserved for options
    int bdr_attr;   // optional/debug/support
    int rsrv; // alignment/expansion
  };

  inline int AppendBCVectorPayload(mfem::Vector &dst,
                                   const mfem::Vector &src)
  {
    const int offset = dst.Size();
    const int n = src.Size();
    mfem::Vector old(dst);
    dst.SetSize(offset + n);
    
    for (int i = 0; i < offset; ++i)
      {
        dst[i] = old[i];
      }
    for(int i = 0;i < n;i++)
      {
        dst[i+offset] = src[i];
      }

    return offset;
  }

} // namespace Theseus
