// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "mfem.hpp"
#include "StabilityEstimate.hpp"
#include "AxisymmetricGeometry.hpp"
#include "dgsem_cache.hpp"
#include "ModalBasis.hpp"
#include "timer.hpp"

namespace Theseus {

  // TODO: Not complete as written, after this, callsites need to set up
  //       Boundary Face data structures with a separate call.  Need to fix.
  template<typename CacheT>
  void GetOperatorCache(mfem::FiniteElementSpace *fes, CacheT *cache)
  {
    GetDiscretizationInfo(fes, cache);
    {
      Theseus::ScopedTimer timer("SetupRestrictions");
      SetupRestrictions(fes, cache);
    }
    SetupVolumeMarkers(fes, cache);
    {
      Theseus::ScopedTimer timer("SetupGeometricTerms");
      SetupGeometricTerms(fes, cache);
    }
    // AssembleBoundaryFaceGeometryTerms(fes, cache);
    // TODO: Move these to where the caches are created and validated
    // MFEM_VERIFY(nfaces == cache.num_interior_faces, "restriction faces != cached interior faces");
    // MFEM_VERIFY(cache.face_normals.Size() == nfaces*nfp*dim, "normals size mismatch");
    // MFEM_VERIFY(cache.face_wt_minus.Size() == nfaces*nfp, "w_minus size mismatch");
    // MFEM_VERIFY(cache.face_wt_plus.Size()  == nfaces*nfp, "w_plus size mismatch");
  }

  template<typename CacheT>
  void GetDiscretizationInfo(mfem::FiniteElementSpace *fes, CacheT *cache)
  {

    MFEM_VERIFY(fes, "fes must be set");
    mfem::Mesh *mesh = fes->GetMesh();
    MFEM_VERIFY(mesh, "mesh must be set");
    const int p = fes->GetFE(0)->GetOrder();
    const int dim = mesh->SpaceDimension();
    const int Np = p + 1; // num 1d quadrature points
    const int Np_x = Np;
    const int Np_y = dim > 1 ? Np : 1;
    const int Np_z = dim > 2 ? Np : 1;
    const int ne = fes->GetNE();
    const int num_dofs_per_eqn_per_element = fes->GetFE(0)->GetDof();
    const int num_eqns = fes->GetVDim();

    cache->p = p;
    cache->Np = Np;
    cache->dim = dim;
    cache->Np_x = Np_x;
    cache->Np_y = Np_y;
    cache->Np_z = Np_z;
    cache->num_elements = ne;
    cache->ndof_scalar_el = num_dofs_per_eqn_per_element;
    cache->num_equations = num_eqns;
  }

  template<typename CacheT>
  void SetupRestrictions(mfem::FiniteElementSpace *fes, CacheT *cache)
  {
    auto *pfes = dynamic_cast<mfem::ParFiniteElementSpace*>(fes);
    MFEM_VERIFY(pfes, "Restriction setup requires ParFiniteElementSpace");
    cache->restr_v = fes->GetElementRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC);
    cache->restr_f = pfes->GetFaceRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC,
                                              mfem::FaceType::Interior,
                                              mfem::L2FaceValues::DoubleValued);
    cache->restr_b = pfes->GetFaceRestriction(mfem::ElementDofOrdering::LEXICOGRAPHIC,
                                              mfem::FaceType::Boundary,
                                              mfem::L2FaceValues::SingleValued);
  }

  // Set up and populate elJac, elMetric, D, Dhat, Dhat2
  // Face normals, and weights
  template<typename CacheT>
  void SetupGeometricTerms(mfem::FiniteElementSpace *fes, CacheT *cache)
  {
    const int nelem = cache->num_elements;
    const int p = cache->p;
    const int Np = cache->Np;
    const int dim = cache->dim;
    const int Np_x = Np;
    const int Np_y = dim > 1 ? Np : 1;
    const int Np_z = dim > 2 ? Np : 1;
    const int neq = cache->num_equations;
    mfem::Mesh *mesh = fes->GetMesh();

    // Build integration rules
    const int IntegrationOrder = 2 * Np_x - 3;
    cache->ir = &cache->GLIntRules.Get(mfem::Geometry::SEGMENT, IntegrationOrder);
    auto vol_topo = (dim == 1 ? mfem::Geometry::SEGMENT :
                     (dim == 2 ? mfem::Geometry::SQUARE : mfem::Geometry::CUBE));
    auto face_topo = (dim == 1 ? mfem::Geometry::POINT :
                      (dim == 2 ? mfem::Geometry::SEGMENT : mfem::Geometry::SQUARE));

    cache->ir_face = &cache->GLIntRules.Get(face_topo, IntegrationOrder);
    cache->ir_vol = &cache->GLIntRules.Get(vol_topo, IntegrationOrder);

    MFEM_ASSERT(cache->ir->GetNPoints() == Np_x, "");
    MFEM_ASSERT(cache->ir_vol->GetNPoints() == Np_x*Np_y*Np_z, "");

    // Populate element Jacobian determinant and metric terms
    cache->elJac.SetSize(Np_x*Np_y*Np_z*nelem);
    cache->elMetric.SetSize(dim*dim*Np_x*Np_y*Np_z*nelem);
    cache->elQuadratureWeights.SetSize(Np_x*Np_y*Np_z*nelem);
    cache->elRadius.SetSize(dim > AxisymmetricGeometry::radial_coordinate ?
                            Np_x*Np_y*Np_z*nelem : 0);
    for (int i = 0; i < nelem; i++)
      {
        mfem::ElementTransformation *T = fes->GetElementTransformation(i);
        assert(T->ElementNo == i);
        AssembleElementVolumeGeometricTerms(*T, cache);
      }

    // Set up derivative operators
    mfem::DenseMatrix D_T, Dhat_T, Dhat2_T;
    D_T.SetSize(Np_x);
    Dhat_T.SetSize(Np_x);
    Dhat2_T.SetSize(Np_x);

    mfem::Vector wBary(Np_x);
    wBary = 1.0;

    for (int i = 1; i < Np_x; i++)
      {
        for (int j = 0; j < i; j++)
          {
            wBary(j) *= (cache->ir->IntPoint(j).x - cache->ir->IntPoint(i).x);
            wBary(i) *= (cache->ir->IntPoint(i).x - cache->ir->IntPoint(j).x);
          }
      }

    wBary.Reciprocal();
    D_T = 0.0;
    for (int iL = 0; iL < Np_x; iL++)
      {
        for (int i = 0; i < Np_x; i++)
          {
            if (iL != i)
              {
                D_T(i, iL) = wBary(iL) / wBary(i) / (cache->ir->IntPoint(i).x - cache->ir->IntPoint(iL).x);
                D_T(i, i) -= D_T(i, iL);
              }
          }
      }

    Dhat_T = D_T;
    Dhat_T(0, 0) += 1.0 / cache->ir->IntPoint(0).weight;
    Dhat_T(Np - 1, Np - 1) -= 1.0 / cache->ir->IntPoint(Np - 1).weight;
    Dhat_T.Transpose();

    Dhat2_T = D_T;
    Dhat2_T *= 2.0;
    Dhat2_T(0, 0) += 1.0 / cache->ir->IntPoint(0).weight;
    Dhat2_T(Np - 1, Np - 1) -= 1.0 / cache->ir->IntPoint(Np - 1).weight;

    cache->stabilityAdvectionScale = ReferenceAdvectionSpectralScale(p);
    cache->stabilityDiffusionScale = ReferenceBR1DiffusionSpectralScale(p);
    const mfem::real_t endpoint_lift =
      1.0 / cache->ir->IntPoint(0).weight;
    cache->stabilitySurfaceScale =
      cache->stabilityAdvectionScale / endpoint_lift;

    Dhat2_T.Transpose();
    D_T.Transpose();

    // Just copy D_T, Dhat_T, and Dhat2_T
    cache->D.SetSize(Np_x*Np_x);
    cache->Dhat.SetSize(Np_x*Np_x);
    cache->Dhat2.SetSize(Np_x*Np_x);
    std::memcpy(cache->D.HostWrite(),     D_T.Data(),     sizeof(mfem::real_t)*Np_x*Np_x);
    std::memcpy(cache->Dhat.HostWrite(),  Dhat_T.Data(),  sizeof(mfem::real_t)*Np_x*Np_x);
    std::memcpy(cache->Dhat2.HostWrite(), Dhat2_T.Data(), sizeof(mfem::real_t)*Np_x*Np_x);

#ifdef POINT_PARALLEL_VOLUME
      cache->elWaveSpeed.SetSize(nelem * cache->ndof_scalar_el);
#else
    cache->elWaveSpeed.SetSize(nelem);
#endif
    cache->elWaveSpeed = 0.0;
    cache->elWaveSpeed.UseDevice();
    cache->elWaveSpeed.Read();

    cache->elJac.UseDevice();
    cache->elMetric.UseDevice();
    cache->elRadius.UseDevice();
    cache->D.UseDevice();
    cache->Dhat.UseDevice();
    cache->Dhat2.UseDevice();
    cache->elJac.Read();
    cache->elMetric.Read();
    cache->elRadius.Read();
    cache->D.Read();
    cache->Dhat.Read();
    cache->Dhat2.Read();

    // Set up data for faces
    const int nfp = cache->ir_face->GetNPoints();
    cache->num_face_points = nfp;

    const int nfaces_restr = cache->restr_f->Height() / (nfp * neq * 2);
    cache->num_interior_faces = nfaces_restr;
    MFEM_VERIFY(nfaces_restr > 0, "nfaces_restr is 0");

    AssembleInteriorFaceGeometryTerms(fes, cache);

    cache->face_normals.UseDevice();
    cache->face_wt_minus.UseDevice();
    cache->face_wt_plus.UseDevice();
    cache->face_radius.UseDevice();
    cache->face_normals.Read();
    cache->face_wt_minus.Read();
    cache->face_wt_plus.Read();
    cache->face_radius.Read();

    cache->ifWaveSpeed.SetSize(cache->num_interior_faces * nfp);
    cache->ifWaveSpeed = 0.0;
    cache->ifWaveSpeed.UseDevice();
    cache->ifWaveSpeed.Read();
  }


  template<typename CacheT>
  void SetupVolumeMarkers(mfem::FiniteElementSpace *fes, CacheT *cache)
  {
    mfem::Mesh *mesh = fes->GetMesh();

    cache->num_attr = mesh->attributes.Size() ? mesh->attributes.Max() : 0;
    cache->vol_attr_marker.SetSize(cache->num_attr);
    cache->vol_attr_marker = 1; // process everything

    cache->domain_attr_marker.SetSize(cache->num_attr);
    cache->domain_attr_marker = 1; // process everything

    // ---- 2) Per-element attribute id array -----------------------------------
    const int ne = mesh->GetNE();
    cache->elem_attr.SetSize(ne);
    for (int e = 0; e < ne; ++e)
      {
        const int attr = mesh->GetAttribute(e); // 1-based
        cache->elem_attr[e] = attr;
      }

    // Optional host-side sanity check (cheap, catches bad markers early):
    if (cache->num_attr > 0)
      {
        for (int e = 0; e < ne; ++e)
          {
            const int a = cache->elem_attr[e];
            MFEM_VERIFY(a >= 1 && a <= cache->num_attr,
                        "element attribute out of range: attr=" << a
                        << " num_attr=" << cache->num_attr);
          }
      }

    cache->elem_attr.UseDevice();
    cache->vol_attr_marker.UseDevice();
    cache->domain_attr_marker.UseDevice();
    cache->elem_attr.Read();
    cache->vol_attr_marker.Read();
    cache->domain_attr_marker.Read();
  }


  template <typename CacheT>
  void BuildBoundaryFacePermutationMap(mfem::ParMesh *pmesh, CacheT *cache)
  {
    MFEM_VERIFY(cache->ir_face, "cache->ir_face must exist");

    auto &bnd_faces = pmesh->GetFaceIndices(mfem::FaceType::Boundary);
    const int nbnd_faces = bnd_faces.Size();
    const int nfp = cache->ir_face->GetNPoints();

    cache->inv_fp_map_bnd.SetSize(nbnd_faces * nfp);

    for (int fslot = 0; fslot < nbnd_faces; ++fslot)
      {
        for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
          {
            const int fp_perm = cache->fqs_bnd->GetPermutedIndex(fslot, fp_restr);
            cache->inv_fp_map_bnd[fslot * nfp + fp_perm] = fp_restr;
          }
      }
  }

  template <typename CacheT>
  void BuildBoundaryFaceToMarkerMap(mfem::ParMesh *pmesh,
                                    const std::vector<mfem::Array<int>> &bdr_marker_vector,
                                    CacheT *cache)
  {
    auto &bnd_faces = pmesh->GetFaceIndices(mfem::FaceType::Boundary);
    const auto &bnd_face_attr = pmesh->GetBdrFaceAttributes();

    const int nbnd_faces = bnd_faces.Size();

    MFEM_VERIFY(bnd_face_attr.Size() == nbnd_faces,
                "Expected compact boundary-face attribute array");

    cache->bnd_attr.SetSize(nbnd_faces);
    cache->bnd_marker_index.SetSize(nbnd_faces);

    for (int fslot = 0; fslot < nbnd_faces; ++fslot)
      {
        const int attr = bnd_face_attr[fslot];
        cache->bnd_attr[fslot] = attr;
        cache->bnd_marker_index[fslot] = -1;

        if (attr <= 0) { continue; }

        for (int mindex = 0; mindex < (int)bdr_marker_vector.size(); ++mindex)
          {
            const auto &marker = bdr_marker_vector[mindex];
            MFEM_VERIFY(attr-1 < marker.Size(), "boundary attribute out of marker range");

            if (marker[attr-1])
              {
                cache->bnd_marker_index[fslot] = mindex;
                break;
              }
          }
      }
  }

  inline void BuildBoundaryFaceToBEMap(mfem::ParMesh *pmesh,
                                       mfem::Array<int> &face_to_be)
  {
    const int nfaces = pmesh->GetNumFaces();
    face_to_be.SetSize(nfaces);
    face_to_be = -1;

    for (int be = 0; be < pmesh->GetNBE(); ++be)
      {
        const int face_id = pmesh->GetBdrElementFaceIndex(be);
        // const int face_id = pmesh->GetBdrFace(be);
        MFEM_VERIFY(face_id >= 0 && face_id < nfaces, "bad boundary face id");
        MFEM_VERIFY(face_to_be[face_id] < 0, "duplicate boundary element for face");
        face_to_be[face_id] = be;
      }
  }

  template<typename CacheT>
  void AssembleBoundaryFaceGeometryTerms(mfem::FiniteElementSpace *fes,
                                         const std::vector<mfem::Array<int>> &bdr_marker_vector,
                                         CacheT *cache)
  {
    auto *mesh  = fes->GetMesh();
    auto *pmesh = dynamic_cast<mfem::ParMesh*>(mesh);
    auto *pfes  = dynamic_cast<mfem::ParFiniteElementSpace*>(fes);

    MFEM_VERIFY(pmesh, "need ParMesh");
    MFEM_VERIFY(pfes,  "need ParFiniteElementSpace");
    MFEM_VERIFY(cache, "need cache");
    MFEM_VERIFY(cache->ir_face, "cache->ir_face must exist");
    MFEM_VERIFY(cache->ir, "cache->ir must exist");

    cache->fqs_bnd.reset(new mfem::FaceQuadratureSpace(*mesh, *cache->ir_face,
                                                       mfem::FaceType::Boundary));

    const int dim = mesh->Dimension();
    const int nfp = cache->ir_face->GetNPoints();

    auto &bnd_faces = pmesh->GetFaceIndices(mfem::FaceType::Boundary);
    const int nbnd_faces = bnd_faces.Size();

    cache->bndWaveSpeed.SetSize(nbnd_faces*nfp);
    cache->bndWaveSpeed = 0.0;
    cache->bndWaveSpeed.Read();

    // 0. Get a restriction-face-to-bnd-element mapping
    mfem::Array<int> face_to_be;
    BuildBoundaryFaceToBEMap(pmesh, face_to_be);

    // 1. Permutation map
    BuildBoundaryFacePermutationMap(pmesh, cache);

    // 2. BC mapping
    BuildBoundaryFaceToMarkerMap(pmesh, bdr_marker_vector, cache);

    // 3. Geometry arrays
    cache->bnd_normals.SetSize(nbnd_faces * nfp * dim);
    cache->bnd_wt.SetSize(nbnd_faces * nfp);
    cache->bnd_xyz.SetSize(nbnd_faces * nfp * dim);
    cache->bnd_radius.SetSize(dim > AxisymmetricGeometry::radial_coordinate ?
                              nbnd_faces * nfp : 0);

    mfem::real_t *nor_d = cache->bnd_normals.HostWrite();
    mfem::real_t *wt_d  = cache->bnd_wt.HostWrite();
    mfem::real_t *xyz_d = cache->bnd_xyz.HostWrite();
    mfem::real_t *rad_d = cache->bnd_radius.Size() > 0 ?
                          cache->bnd_radius.HostWrite() : nullptr;

    const mfem::real_t w0 = cache->ir->IntPoint(0).weight;

    mfem::Vector nor(dim);
    mfem::Vector phys(dim);

    auto store = [&](int fslot, int fp_restr,
                     const mfem::Vector &nor,
                     const mfem::Vector &phys,
                     mfem::real_t inv_wJ1)
    {
      const int base_scl = fslot * nfp + fp_restr;
      const int base_vec = base_scl * dim;

      for (int d = 0; d < dim; ++d)
        {
          nor_d[base_vec + d] = nor(d);
          xyz_d[base_vec + d] = phys(d);
        }

      wt_d[base_scl] = inv_wJ1;
      if (rad_d)
        {
          const mfem::real_t radius =
            AxisymmetricGeometry::Radius(phys.GetData());
          rad_d[base_scl] = cache->axisymmetric ?
            AxisymmetricGeometry::ValidateRadius(
              radius, "boundary face " + std::to_string(fslot) +
              ", point " + std::to_string(fp_restr)) : radius;
        }
    };

    for (int fslot = 0; fslot < nbnd_faces; ++fslot)
      {
        const int face_id = bnd_faces[fslot];
        const int be_match = face_to_be[face_id];
        // Map boundary face slot -> boundary element index.
        // We need boundary-element index for GetBdrFaceTransformations(be).
        MFEM_VERIFY(be_match >= 0, "Could not find boundary element for boundary face");
        auto *tr = mesh->GetBdrFaceTransformations(be_match);
        MFEM_VERIFY(tr, "expected boundary face transformation");

        for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
          {
            const int fp_geom = cache->inv_fp_map_bnd[fslot * nfp + fp_restr];
            const mfem::IntegrationPoint &ip = cache->ir_face->IntPoint(fp_geom);
            tr->SetAllIntPoints(&ip);

            const mfem::real_t J1 = tr->GetElement1Transformation().Weight();
            if (dim == 1)
              {
                nor(0) = (tr->GetElement1IntPoint().x - 0.5) * 2.0;
              }
            else
              {
                mfem::CalcOrtho(tr->Jacobian(), nor);
              }
            tr->Transform(ip, phys);
            store(fslot, fp_restr, nor, phys, 1.0 / (w0 * J1));
          }
      }
    cache->bnd_xyz.UseDevice();
    cache->bnd_radius.UseDevice();
    cache->bnd_xyz.Read();
    cache->bnd_radius.Read();
  }

  // Builds element-specific Jac/Metric and stuffs into cache.elJac, cache.elMetric
  template<typename CacheT>
  void AssembleElementVolumeGeometricTerms(mfem::ElementTransformation &Tr, CacheT *cache)
  {

    mfem::real_t *Jinv_h = cache->elJac.HostWrite();
    mfem::real_t *Met_h  = cache->elMetric.HostWrite();
    mfem::real_t *qWgts_h = cache->elQuadratureWeights.HostWrite();
    mfem::real_t *radius_h = cache->elRadius.Size() > 0 ?
                             cache->elRadius.HostWrite() : nullptr;

    int dim = cache->dim;
    mfem::Vector metric1(dim);
    mfem::Vector physical(dim);
    const int e = Tr.ElementNo;
    const int nq = cache->Np_x * cache->Np_y * cache->Np_z;

    for (int q = 0; q < nq; ++q)
      {
        const mfem::IntegrationPoint &ip = cache->ir_vol->IntPoint(q);
        Tr.SetIntPoint(&ip);
        const mfem::real_t J = Tr.Weight();
        Jinv_h[e*nq + q] = J;
        qWgts_h[e*nq + q] = J * ip.weight;
        if (radius_h)
          {
            Tr.Transform(ip, physical);
            const mfem::real_t radius =
              AxisymmetricGeometry::Radius(physical.GetData());
            radius_h[e*nq + q] = cache->axisymmetric ?
              AxisymmetricGeometry::ValidateRadius(
                radius, "element " + std::to_string(e) +
                ", point " + std::to_string(q)) : radius;
          }
        const mfem::DenseMatrix &adj = Tr.AdjugateJacobian();
        for (int dir = 0; dir < dim; ++dir)
          {
            adj.GetRow(dir, metric1);  // metric1.Size() == dim

            for (int d = 0; d < dim; ++d)
              {
                const int idxM = (((e*nq + q)*dim + dir)*dim + d);
                Met_h[idxM] = metric1(d);
              }
          }
      }
  }

  template<typename CacheT>
  void AssembleInteriorFaceGeometryTerms(mfem::FiniteElementSpace *fes, CacheT *cache)
  {
    auto *mesh = fes->GetMesh();
    auto *pmesh = dynamic_cast<mfem::ParMesh*>(mesh);
    auto *pfes = dynamic_cast<mfem::ParFiniteElementSpace*>(fes);
    cache->fqs_int.reset(new mfem::FaceQuadratureSpace(*mesh, *cache->ir_face,
                                                       mfem::FaceType::Interior));
    MFEM_VERIFY(pfes, "need ParFiniteElementSpace");

    const int dim = mesh->Dimension();
    const int neq = pfes->GetVDim();
    const int nfp = cache->ir_face->GetNPoints();

    auto &int_faces = pmesh->GetFaceIndices(mfem::FaceType::Interior);
    const int ninterior_faces = int_faces.Size();

    cache->inv_fp_map.SetSize(ninterior_faces * nfp);
    for (int face_slot = 0; face_slot < ninterior_faces; ++face_slot)
      {
        for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
          {
            int fp_perm = cache->fqs_int->GetPermutedIndex(face_slot, fp_restr);
            cache->inv_fp_map[face_slot*nfp + fp_perm] = fp_restr;
          }
      }

    cache->face_normals.SetSize(ninterior_faces * nfp * dim);
    cache->face_wt_minus.SetSize(ninterior_faces * nfp);
    cache->face_wt_plus.SetSize(ninterior_faces * nfp);
    cache->face_radius.SetSize(
      dim > AxisymmetricGeometry::radial_coordinate ?
      ninterior_faces * nfp : 0);

    mfem::real_t *nor_d  = cache->face_normals.HostWrite();
    mfem::real_t *inv1_d = cache->face_wt_minus.HostWrite();
    mfem::real_t *inv2_d = cache->face_wt_plus.HostWrite();
    mfem::real_t *radius_d = cache->face_radius.Size() > 0 ?
                             cache->face_radius.HostWrite() : nullptr;
    const mfem::real_t w0 = cache->ir->IntPoint(0).weight;

    auto store = [&](int fslot, int fp, const mfem::Vector &nor,
                     const mfem::Vector &physical,
                     mfem::real_t inv_wJ1, mfem::real_t inv_wJ2)
    {
      const int nbase = (fslot * nfp + fp) * dim;
      for (int d = 0; d < dim; ++d) { nor_d[nbase + d] = nor(d); }
      inv1_d[fslot * nfp + fp] = inv_wJ1;
      inv2_d[fslot * nfp + fp] = inv_wJ2;
      if (radius_d)
        {
          const mfem::real_t radius =
            AxisymmetricGeometry::Radius(physical.GetData());
          radius_d[fslot*nfp + fp] = cache->axisymmetric ?
            AxisymmetricGeometry::ValidateRadius(
              radius, "interior face " + std::to_string(fslot) +
              ", point " + std::to_string(fp)) : radius;
        }
    };

    mfem::Vector nor(dim);
    mfem::Vector physical(dim);
    // The order of faces in GetFaceIndices(FaceType::Interior) *must*
    // match the order of the faces in the interior face restriction
    // operator face slots.
    for (int fslot = 0; fslot < ninterior_faces; ++fslot)
      {
        const int face_id = int_faces[fslot];
        // bool face_is_flipped = false;
        // for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
        //   {
        //     const int fp_geom = cache->MapFp(fslot, fp_restr);// <-- critical
        //     if (fp_geom != fp_restr){
        //       face_is_flipped = true;
        //     }
        //   }
        auto *tr = mesh->GetInteriorFaceTransformations(face_id);
        if (tr){ // Do interior face caching
          //          MFEM_VERIFY(tr, "expected interior face");
          for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
            {
              const int fp_geom = cache->MapFp(fslot, fp_restr);// <-- critical
              const mfem::IntegrationPoint &ip = cache->ir_face->IntPoint(fp_geom);
              tr->SetAllIntPoints(&ip);

              const mfem::real_t J1 = tr->GetElement1Transformation().Weight();
              const mfem::real_t J2 = tr->GetElement2Transformation().Weight();

              if (dim == 1) { nor(0) = (tr->GetElement1IntPoint().x - 0.5)*2.0; }
              else          { mfem::CalcOrtho(tr->Jacobian(), nor); }
              tr->Transform(ip, physical);

              //const mfem::real_t fac = face_is_flipped ? -1.0 : 1.0;
              const mfem::real_t fac = 1.0;
              store(fslot, fp_restr, nor, physical,
                    fac/(w0*J1), fac/(w0*J2));
            }
          continue;
        } // Internal face processing
        {
          auto *sh_tr = pmesh->GetSharedFaceTransformationsByLocalIndex(face_id, true);
          MFEM_VERIFY(sh_tr, "expected shared face");
          for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
            {
              const int fp_geom = cache->MapFp(fslot, fp_restr);// <-- critical
              const mfem::IntegrationPoint &ip = cache->ir_face->IntPoint(fp_geom);
              sh_tr->SetAllIntPoints(&ip);

              const mfem::real_t J1 = sh_tr->GetElement1Transformation().Weight();
              const mfem::real_t J2 = sh_tr->GetElement2Transformation().Weight();

              if (dim == 1) { nor(0) = (sh_tr->GetElement1IntPoint().x - 0.5)*2.0; }
              else          { mfem::CalcOrtho(sh_tr->Jacobian(), nor); }
              sh_tr->Transform(ip, physical);

              //const mfem::real_t fac = face_is_flipped ? -1.0 : 1.0;
              const mfem::real_t fac1 = 1.0;
              const mfem::real_t fac2 = 0.0;
              store(fslot, fp_restr, nor, physical,
                    fac1/(w0*J1), fac2/(w0*J2));
            }
        } // Shared face processing
      } // Interior face processing
  }

  template<typename CacheT>
  void ComputeSubcellMetrics(mfem::FiniteElementSpace *fes, CacheT *cache)
  {
    MFEM_VERIFY(fes != nullptr, "A finite element space is required.");
    const int dim = cache->dim;
    const int ne = cache->num_elements;
    const int Np_x = cache->Np_x;
    const int Np_y = cache->Np_y;
    const int Np_z = cache->Np_z;
    const int nq = cache->ndof_scalar_el;
    const int n_metric_xi = (Np_x + 1) * Np_y * Np_z;
    const int n_metric_eta = Np_x * (Np_y + 1) * Np_z;
    const int n_metric_zeta = Np_x * Np_y * (Np_z + 1);

    cache->subcellWeights.SetSize(Np_x);
    mfem::real_t *wgts = cache->subcellWeights.HostWrite();
    for (int i = 0;i < Np_x;i++){
      wgts[i] = cache->ir->IntPoint(i).weight;
    }

    cache->subcellMetricXi.SetSize(n_metric_xi*dim*ne);
    if (dim > 1)
      cache->subcellMetricEta.SetSize(n_metric_eta*dim*ne);
    if (dim > 2)
      cache->subcellMetricZeta.SetSize(n_metric_zeta*dim*ne);

    const mfem::real_t *metric = cache->elMetric.Read();
    const mfem::real_t *D = cache->D.Read();
    const mfem::real_t *weights = cache->subcellWeights.Read();
    mfem::real_t *xi = cache->subcellMetricXi.Write();

    const int xi_work = ne*Np_z*Np_y*(Np_x - 1)*dim;
    mfem::forall(xi_work, [=] MFEM_HOST_DEVICE (int tid) {
      const int d = tid % dim;
      int item = tid / dim;
      const int i = item % (Np_x - 1) + 1;
      item /= Np_x - 1;
      const int j = item % Np_y;
      item /= Np_y;
      const int k = item % Np_z;
      const int el = item / Np_z;
      const int line = k*Np_y*Np_x + j*Np_x;
      mfem::real_t normal = metric[((el*nq + line)*dim)*dim + d];
      for (int l = 0; l < i; ++l)
        {
          mfem::real_t sum = 0.0;
          for (int m = 0; m < Np_x; ++m)
            {
              mfem::real_t contribution =
                metric[((el*nq + line + m)*dim)*dim + d];
              contribution *= D[l*Np_x + m];
              sum += contribution;
            }
          sum *= weights[l];
          normal += sum;
        }
      xi[((el*n_metric_xi + line + i)*dim) + d] = normal;
    });

    if (dim > 1)
      {
        mfem::real_t *eta = cache->subcellMetricEta.Write();
        const int eta_work = ne*Np_z*Np_x*(Np_y - 1)*dim;
        mfem::forall(eta_work, [=] MFEM_HOST_DEVICE (int tid) {
          const int d = tid % dim;
          int item = tid / dim;
          const int j = item % (Np_y - 1) + 1;
          item /= Np_y - 1;
          const int i = item % Np_x;
          item /= Np_x;
          const int k = item % Np_z;
          const int el = item / Np_z;
          const int line = k*Np_y*Np_x + i;
          mfem::real_t normal = metric[((el*nq + line)*dim + 1)*dim + d];
          for (int l = 0; l < j; ++l)
            {
              mfem::real_t sum = 0.0;
              for (int m = 0; m < Np_y; ++m)
                {
                  mfem::real_t contribution =
                    metric[((el*nq + line + m*Np_x)*dim + 1)*dim + d];
                  contribution *= D[l*Np_x + m];
                  sum += contribution;
                }
              sum *= weights[l];
              normal += sum;
            }
          eta[((el*n_metric_eta + k*Np_y*Np_x + j*Np_x + i)*dim) + d] = normal;
        });
      }

    if (dim > 2)
      {
        mfem::real_t *zeta = cache->subcellMetricZeta.Write();
        const int zeta_work = ne*Np_y*Np_x*(Np_z - 1)*dim;
        mfem::forall(zeta_work, [=] MFEM_HOST_DEVICE (int tid) {
          const int d = tid % dim;
          int item = tid / dim;
          const int k = item % (Np_z - 1) + 1;
          item /= Np_z - 1;
          const int i = item % Np_x;
          item /= Np_x;
          const int j = item % Np_y;
          const int el = item / Np_y;
          const int line = j*Np_x + i;
          mfem::real_t normal = metric[((el*nq + line)*dim + 2)*dim + d];
          for (int l = 0; l < k; ++l)
            {
              mfem::real_t sum = 0.0;
              for (int m = 0; m < Np_z; ++m)
                {
                  mfem::real_t contribution = metric[
                    ((el*nq + line + m*Np_y*Np_x)*dim + 2)*dim + d];
                  contribution *= D[l*Np_x + m];
                  sum += contribution;
                }
              sum *= weights[l];
              normal += sum;
            }
          zeta[((el*n_metric_zeta + k*Np_y*Np_x + line)*dim) + d] = normal;
        });
      }
  }

  template<typename CacheT, typename DeviceCacheT>
  void GetDeviceCache(CacheT &cache, DeviceCacheT &device_cache)
  {
    // Fixed data items
    // - Discretization parameters:
    device_cache.ndof_scalar_el = cache.ndof_scalar_el;
    device_cache.num_attr = cache.num_attr;
    device_cache.attr_marker_d = cache.vol_attr_marker.Read();
    device_cache.elem_attr_d = cache.elem_attr.Read();
    device_cache.num_face_points = cache.num_face_points;
    device_cache.p = cache.p;
    device_cache.dim = cache.dim;
    device_cache.Np = cache.Np;
    device_cache.Np_x = cache.Np_x;
    device_cache.Np_y = cache.Np_y;
    device_cache.Np_z = cache.Np_z;
    device_cache.num_elements = cache.num_elements;
    device_cache.num_equations = cache.num_equations;
    device_cache.axisymmetric = cache.axisymmetric;

    // - Volume element data
    device_cache.elJac_d = cache.elJac.Read();
    device_cache.elMetric_d = cache.elMetric.Read();
    device_cache.D_d = cache.D.Read();
    device_cache.Dhat_d = cache.Dhat.Read();
    device_cache.Dhat2_d = cache.Dhat2.Read();
    device_cache.elQWgts_d = cache.elQuadratureWeights.Read();
    device_cache.elRadius_d = cache.elRadius.Read();

    // - Interior faces (including remote)
    device_cache.nor_d = cache.face_normals.Read();
    device_cache.fw_minus_d = cache.face_wt_minus.Read();
    device_cache.fw_plus_d = cache.face_wt_plus.Read();
    device_cache.face_radius_d = cache.face_radius.Read();

    // - Boundary faces
    device_cache.bnd_nor_d = cache.bnd_normals.Read();
    device_cache.bnd_wt_d = cache.bnd_wt.Read();
    device_cache.bnd_radius_d = cache.bnd_radius.Read();
    device_cache.bnd_marker_index_d = cache.bnd_marker_index.Read();
    device_cache.bnd_marker_to_bc_descr_d = cache.bnd_marker_to_bc_descr.Read();
    device_cache.bc_scalar_d = cache.bc_scalar_data.Read();
    device_cache.bc_vector_d = cache.bc_vector_data.Read();
    device_cache.bc_descr_d = cache.bc_descriptors.Read();

    // Updated every step by the compute device
    device_cache.elWaveSpeed_d = cache.elWaveSpeed.ReadWrite();
    device_cache.ifWaveSpeed_d = cache.ifWaveSpeed.ReadWrite();
    device_cache.bndWaveSpeed_d = cache.bndWaveSpeed.ReadWrite();

    // POD gas model
    device_cache.gas = cache.gas.to_device(cache);
    device_cache.iflux = cache.iflux;

#ifdef SUBCELL_FV_BLENDING
    device_cache.subcell_metric_xi_d = cache.subcellMetricXi.Read();
    device_cache.subcell_metric_eta_d = cache.subcellMetricEta.Read();
    device_cache.subcell_metric_zeta_d = cache.subcellMetricZeta.Read();
    device_cache.subcell_weights_d = cache.subcellWeights.Read();
#endif

  }

  template<typename CacheT>
  bool AxisBoundaryGeometryIsValid(const CacheT &cache)
  {
    const int points_per_face = cache.num_face_points;
    const int boundary_faces = cache.bnd_marker_index.Size();
    const int *marker_index = cache.bnd_marker_index.HostRead();
    const Theseus::BCDescriptor *descriptors =
      cache.bc_descriptors.HostRead();
    const mfem::real_t *radius = cache.bnd_radius.HostRead();
    for (int face = 0; face < boundary_faces; ++face)
      {
        const int descriptor_index = marker_index[face];
        if (descriptor_index < 0 ||
            descriptor_index >= cache.bc_descriptors.Size() ||
            descriptors[descriptor_index].type !=
              int(Theseus::BCType::Axis))
          {
            continue;
          }
        if (cache.bnd_radius.Size() != boundary_faces*points_per_face)
          {
            return false;
          }
        for (int point = 0; point < points_per_face; ++point)
          {
            const mfem::real_t point_radius =
              radius[face*points_per_face + point];
            if (point_radius > AxisymmetricGeometry::radius_tolerance)
              {
                return false;
              }
          }
      }
    return true;
  }

  template<typename CacheT>
  void ValidateAxisBoundaryGeometry(const CacheT &cache)
  {
    MFEM_VERIFY(AxisBoundaryGeometryIsValid(cache),
                "an axis boundary quadrature point is not on r=0");
  }

  template<typename CacheT>
  void BuildPerssonDeviceCache(CacheT &c,
                               Prandtl::ModalBasis &modalBasis)
  {
    // std::shared_ptr<Prandtl::ModalBasis> modalBasis;
    // mfem::Vector rho_p, modes, modesM1, modesM2;
    // mfem::Array2D<int> ubdegs;
    // mfem::Array<int> ubdegs_row;
    int dim = c.dim;
    int order = c.p;
    int ndofs = c.ndof_scalar_el;
    int ne = c.num_elements;
    const mfem::Array2D<int> ubdegs(modalBasis.GetPolyDegs());

    c.modal.SetSize(ndofs * ndofs);
    c.keep_M1.SetSize(ndofs);
    c.keep_M2.SetSize(ndofs);
    c.eta.SetSize(ne);

    c.modal.UseDevice();
    c.keep_M1.UseDevice();
    c.keep_M2.UseDevice();
    c.eta.UseDevice();

    auto *modal_h = c.modal.HostWrite();
    auto *m1_h = c.keep_M1.HostWrite();
    auto *m2_h = c.keep_M2.HostWrite();

    mfem::Vector nodal(ndofs), modes(ndofs);

    for (int q = 0; q < ndofs; ++q)
      {
        nodal = 0.0;
        nodal(q) = 1.0;

        modalBasis.ComputeModes(nodal, modes);

        for (int m = 0; m < ndofs; ++m)
          {
            // Store the transform transposed.  CheckIndicatorSmoothness assigns
            // adjacent modal rows to adjacent accelerator threads, so this layout
            // turns their coefficient reads into contiguous transactions.
            modal_h[q * ndofs + m] = modes(m);
          }
      }

    mfem::Array<int> row;
    for (int m = 0; m < ndofs; ++m)
      {
        ubdegs.GetRow(m, row);

        bool keep1 = true;
        bool keep2 = true;

        for (int d = 0; d < dim; ++d)
          {
            if (row[d] > order - 2)
              {
                keep2 = false;
              }

            if (row[d] > order - 1)
              {
                keep1 = false;
              }
          }

        m1_h[m] = keep1 ? 1.0 : 0.0;
        m2_h[m] = keep2 ? 1.0 : 0.0;
      }

    c.modal_d = c.modal.Read();
    c.keep_M1_d = c.keep_M1.Read();
    c.keep_M2_d = c.keep_M2.Read();
    c.eta_d = c.eta.ReadWrite();

  }

  template<typename CacheT>
  void OutputCacheContents(const CacheT &cache)
  {
    std::cout << "Cache Contents:" << std::endl
              << "p = " << cache.p << std::endl
              << "dim = " << cache.dim << std::endl
              << "num_elements = " << cache.num_elements << std::endl
              << "Np,Np_x,Np_y,Np_z = " << cache.Np << "," << cache.Np_x
              << "," << cache.Np_y << "," << cache.Np_z << std::endl
              << "num_face_points = " << cache.num_face_points << std::endl
              << "num_attr = " << cache.num_attr << std::endl
              << "ndof_scalar_el = " << cache.ndof_scalar_el << std::endl
              << "num_interior_faces = " << cache.num_interior_faces << std::endl;
    MFEM_VERIFY(cache.ir, "IR is not set");
    MFEM_VERIFY(cache.ir_face, "Face IR not set");
    MFEM_VERIFY(cache.ir_vol, "Volume IR not set");
    MFEM_VERIFY(cache.restr_v, "Volume Restriction not set");
    MFEM_VERIFY(cache.restr_f, "Facial Restriction not set");
    MFEM_VERIFY(cache.ndof_scalar_el == cache.Np_x*cache.Np_y*cache.Np_z,
                "Element dof count not equal to num quadrature points.");
    int ds_size = cache.elem_attr.Size();
    MFEM_VERIFY(ds_size > 0, "Elem attr not set");

    ds_size = cache.elWaveSpeed.Size();
#ifdef POINT_PARALLEL_VOLUME
    MFEM_VERIFY(ds_size == cache.num_elements * cache.ndof_scalar_el,
                "Element point wavespeeds missized.");
#else
    MFEM_VERIFY(ds_size == cache.num_elements, "Element wavespeeds missized.");
#endif
    ds_size = cache.ifWaveSpeed.Size();
    MFEM_VERIFY(ds_size == cache.num_interior_faces * cache.num_face_points,
                "Interior-face wavespeeds missized.");
    ds_size = cache.bndWaveSpeed.Size();
    ds_size = cache.elJac.Size();
    MFEM_VERIFY(ds_size > 0, "Element Jacobians not set");
    ds_size = cache.elMetric.Size();
    MFEM_VERIFY(ds_size > 0, "Element Metrics not set");
    ds_size = cache.D.Size();
    MFEM_VERIFY(ds_size > 0, "Deriv operator not set");
    ds_size = cache.Dhat2.Size();
    MFEM_VERIFY(ds_size > 0, "Dhat2 operator not set");
    ds_size = cache.face_normals.Size();
    MFEM_VERIFY(ds_size == cache.num_face_points*cache.num_interior_faces*cache.dim,
                "Inapropriately sized face normals");
    ds_size = cache.face_wt_minus.Size();
    ds_size = cache.face_wt_plus.Size();
    MFEM_VERIFY(ds_size > 0, "Face weights not set.");
  }

}
