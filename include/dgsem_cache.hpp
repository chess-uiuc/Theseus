// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "mfem.hpp"
#include "GasModel.hpp"
#include "LTEGasModel.hpp"
#include "bc_cache_utilities.hpp"

namespace Theseus
{
  struct IntegralMeasures {
    mfem::real_t mass = 0.0;
    mfem::real_t ke = 0.0;
    mfem::real_t en = 0.0;
    mfem::real_t max_press = 0.0;
    mfem::real_t min_press = 0.0;
    mfem::real_t max_temp = 0.0;
    mfem::real_t min_temp = 0.0;
    mfem::real_t max_dens = 0.0;
    mfem::real_t min_dens = 0.0;
  };

  template<typename PhysicsT>
  struct DGSEMOperatorCacheT {
    using OperatorGasModel = typename PhysicsT::GasModel;
    using InviscidFlux = typename PhysicsT::InviscidFlux;
    // constants needed by kernels
    int p = 0;
    int dim = 0;
    int Np = 0;
    int Np_x = 0;
    int Np_y = 0;
    int Np_z = 0;
    int num_attr = 0;
    int num_elements = 0;
    int num_equations = 0;
    int ndof_scalar_el = 0;
    int num_face_points = 0;
    int num_interior_faces = 0;

    // Host Only: Integration rules, operators, restrictions 
    mfem::IntegrationRules GLIntRules{0, mfem::Quadrature1D::GaussLobatto};
    const mfem::IntegrationRule *ir = nullptr;
    const mfem::IntegrationRule *ir_face = nullptr;
    const mfem::IntegrationRule *ir_vol = nullptr;
    const mfem::ElementRestrictionOperator *restr_v = nullptr; // for volume elements
    const mfem::FaceRestriction *restr_f = nullptr; // for interior faces
    const mfem::FaceRestriction *restr_b = nullptr; // for boundary faces
    std::unique_ptr<mfem::FaceQuadratureSpace> fqs_int; // interior faces perm
    std::unique_ptr<mfem::FaceQuadratureSpace> fqs_bnd; // boundary faces perm

    // Aux data for preprocessing
    mfem::Array<int> inv_fp_map;
    mfem::Array<int> inv_fp_map_bnd;

    // Fixed data arrays for use on device
    mfem::Array<int> elem_attr;    // size ne, values are 1-based attributes
    mfem::Array<int> vol_attr_marker;  // size nattr, 0/1
    mfem::Array<int> domain_attr_marker;  // size nattr, 0/1
    mfem::Vector elJac;
    mfem::Vector elMetric;
    mfem::Vector elQuadratureWeights;
    mfem::Vector D;
    mfem::Vector Dhat;
    mfem::Vector Dhat2;
    mfem::Vector face_normals;
    mfem::Vector face_wt_minus;
    mfem::Vector face_wt_plus;

    // Temporaries/aux storage
    mfem::Vector pdudt;
    mfem::Vector sVol;
    mfem::Vector uVol;
    mfem::Vector volAux;
    mfem::Vector rhsVol;
    mfem::Vector uInt;
    mfem::Vector rhsInt;
    mfem::Vector dudtInt;
    mfem::Vector uBnd;
    mfem::Vector rhsBnd;
    mfem::Vector dudtBnd;
    mfem::Vector entropyState;
    mfem::Vector duBnd;
    mfem::Vector duInt;

    std::vector<mfem::Vector> gradVol;
    std::vector<mfem::Vector> pGrad;
    std::vector<mfem::Vector> gradInt;
    std::vector<mfem::Vector> gradBnd;
    
    // Domain boundary device arrays
    mfem::Vector bnd_normals;
    mfem::Vector bnd_wt;
    mfem::Vector bnd_xyz;
    mfem::Vector bnd_radius;
    mfem::Array<int> bnd_attr;
    mfem::Array<int> bnd_marker_index;
    mfem::Array<int> bnd_marker_to_bc_descr;
    mfem::Array<Theseus::BCDescriptor> bc_descriptors;
    mfem::Vector bc_scalar_data;
    mfem::Vector bc_vector_data;

    // Physics parts - used directly on device
    mutable mfem::Vector elWaveSpeed; // size nelements
    mutable mfem::Vector ifWaveSpeed; // size ninterior faces
    mutable mfem::Vector bndWaveSpeed; // size nbnd faces
    OperatorGasModel gas;
    InviscidFlux iflux;
    std::unique_ptr<Theseus::LTETable::Data> lteTableData;

#ifdef SUBCELL_FV_BLENDING
    mfem::Vector subcellMetricXi;
    mfem::Vector subcellMetricEta;
    mfem::Vector subcellMetricZeta;
    mfem::Vector subcellWeights;    
    mfem::Vector indicatorField;
#endif
    // nullptr if subcell blending is OFF
    std::shared_ptr<mfem::ParGridFunction> alpha;

    // Grab the face dof from the restriction (face,point) index
    // This answers: what is the facial dof that corresponds to
    // the facial point index for a given face in the restriction?
    int MapFp(int face_slot, int fp_restr) const
    {
      return fqs_int->GetPermutedIndex(face_slot, fp_restr);
    }
    // And the inverse mapping
    int MapFpInv(int face_slot, int fp_perm) const {
      return inv_fp_map[face_slot*num_face_points + fp_perm];
    }
  };

  template<typename PhysicsT>
  struct DGSEMDeviceCacheT {
    using Gas = typename PhysicsT::GasModel;
    using InviscidFlux = typename PhysicsT::InviscidFlux;

    int p = 0;
    int dim = 0;
    int num_elements = 0;
    int ndof_scalar_el = 0;
    int num_equations = 0;
    int num_face_points = 0;
    int num_interior_faces = 0;
    int Np_x = 0;
    int Np_y = 0;
    int Np_z = 0;
    int Np = 0;
    int num_attr = 0;
    int num_bcs = 0;

    // Volume elements
    const int *elem_attr_d = nullptr;    // size ne, values are 1-based attributes
    const int *attr_marker_d = nullptr;  // size nattr, 0/1
    const mfem::real_t *elJac_d = nullptr;
    const mfem::real_t *elMetric_d = nullptr;
    const mfem::real_t *D_d = nullptr;
    const mfem::real_t *Dhat_d = nullptr;
    const mfem::real_t *Dhat2_d = nullptr;
    const mfem::real_t *elQWgts_d = nullptr;

    // Internal faces
    const mfem::real_t *nor_d = nullptr;
    const mfem::real_t *fw_minus_d = nullptr;
    const mfem::real_t *fw_plus_d = nullptr;

    // Boundary faces
    const mfem::real_t *bnd_nor_d = nullptr;
    const mfem::real_t *bnd_wt_d = nullptr;
    const int *bnd_attr_d = nullptr;
    const int *bnd_marker_index_d = nullptr;
    const Theseus::BCDescriptor *bc_descr_d = nullptr;
    const mfem::real_t *bc_scalar_d = nullptr;
    const mfem::real_t *bc_vector_d = nullptr;
    const int *bnd_marker_to_bc_descr_d = nullptr;

    // Physics parts
    mfem::real_t *elWaveSpeed_d = nullptr;
    mfem::real_t *ifWaveSpeed_d = nullptr;
    mfem::real_t *bndWaveSpeed_d = nullptr; 
    Gas gas;
    InviscidFlux iflux;

#ifdef SUBCELL_FV_BLENDING
    const mfem::real_t *subcell_metric_xi_d = nullptr;
    const mfem::real_t *subcell_metric_eta_d = nullptr;
    const mfem::real_t *subcell_metric_zeta_d = nullptr;
    const mfem::real_t *subcell_weights_d = nullptr;
#endif

    MFEM_HOST_DEVICE inline int iface_idx(int side, int fp, int eq) const
    {
      return ((side*num_equations + eq)*num_face_points + fp);
    }
    MFEM_HOST_DEVICE inline int iface_size() const
    {
      return 2*num_equations*num_face_points;
    }
  };

}
