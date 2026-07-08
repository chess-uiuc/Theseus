// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "mfem.hpp"
#include "dgsem_cache.hpp"
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
    Dhat2_T.Transpose();
    D_T.Transpose();

    // Just copy D_T, Dhat_T, and Dhat2_T
    cache->D.SetSize(Np_x*Np_x);
    cache->Dhat.SetSize(Np_x*Np_x);
    cache->Dhat2.SetSize(Np_x*Np_x);
    std::memcpy(cache->D.HostWrite(),     D_T.Data(),     sizeof(mfem::real_t)*Np_x*Np_x);
    std::memcpy(cache->Dhat.HostWrite(),  Dhat_T.Data(),  sizeof(mfem::real_t)*Np_x*Np_x);
    std::memcpy(cache->Dhat2.HostWrite(), Dhat2_T.Data(), sizeof(mfem::real_t)*Np_x*Np_x);

    cache->elWaveSpeed.SetSize(nelem);
    cache->elWaveSpeed = 0.0;
    cache->elWaveSpeed.UseDevice();
    cache->elWaveSpeed.Read();

    cache->elJac.UseDevice();
    cache->elMetric.UseDevice();
    cache->D.UseDevice();
    cache->Dhat.UseDevice();
    cache->Dhat2.UseDevice();
    cache->elJac.Read();
    cache->elMetric.Read();
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
    cache->face_normals.Read();
    cache->face_wt_minus.Read();
    cache->face_wt_plus.Read();

    cache->ifWaveSpeed.SetSize(cache->num_interior_faces);
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
    cache->bnd_xyz.SetSize(nbnd_faces * nfp * dim);     // recommended
    cache->bnd_radius.SetSize(nbnd_faces * nfp);        // optional, useful later
    
    mfem::real_t *nor_d = cache->bnd_normals.HostWrite();
    mfem::real_t *wt_d  = cache->bnd_wt.HostWrite();
    mfem::real_t *xyz_d = cache->bnd_xyz.HostWrite();
    mfem::real_t *rad_d = cache->bnd_radius.HostWrite();

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
      rad_d[base_scl] = (dim > 1) ? phys(1) : 0.0;
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
  }

  // Builds element-specific Jac/Metric and stuffs into cache.elJac, cache.elMetric
  template<typename CacheT>
  void AssembleElementVolumeGeometricTerms(mfem::ElementTransformation &Tr, CacheT *cache)
  {
    
    mfem::real_t *Jinv_h = cache->elJac.HostWrite();
    mfem::real_t *Met_h  = cache->elMetric.HostWrite();
    mfem::real_t *qWgts_h = cache->elQuadratureWeights.HostWrite();

    int dim = cache->dim;
    mfem::Vector metric1(dim);
    const int e = Tr.ElementNo;
    const int nq = cache->Np_x * cache->Np_y * cache->Np_z;
    
    for (int q = 0; q < nq; ++q)
      {
        const mfem::IntegrationPoint &ip = cache->ir_vol->IntPoint(q);
        Tr.SetIntPoint(&ip);
        const mfem::real_t J = Tr.Weight();
        Jinv_h[e*nq + q] = J;
        qWgts_h[e*nq + q] = J * ip.weight;
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
 
    mfem::real_t *nor_d  = cache->face_normals.HostWrite();
    mfem::real_t *inv1_d = cache->face_wt_minus.HostWrite();
    mfem::real_t *inv2_d = cache->face_wt_plus.HostWrite();
    const mfem::real_t w0 = cache->ir->IntPoint(0).weight;
    
    auto store = [&](int fslot, int fp, const mfem::Vector &nor,
                     mfem::real_t inv_wJ1, mfem::real_t inv_wJ2)
    {
      const int nbase = (fslot * nfp + fp) * dim;
      for (int d = 0; d < dim; ++d) { nor_d[nbase + d] = nor(d); }
      inv1_d[fslot * nfp + fp] = inv_wJ1;
      inv2_d[fslot * nfp + fp] = inv_wJ2;
    };
    
    mfem::Vector nor(dim);
    // The order of faces in GetFaceIndices(FaceType::Interior) *must*
    // match the order of the faces in the interior face restriction
    // operator face slots.
    for (int fslot = 0; fslot < ninterior_faces; ++fslot)
      {
        const int face_id = int_faces[fslot];  
        bool face_is_flipped = false;
        for (int fp_restr = 0; fp_restr < nfp; ++fp_restr)
          {
            const int fp_geom = cache->MapFp(fslot, fp_restr);// <-- critical
            if (fp_geom != fp_restr){
              face_is_flipped = true;
            }
          }
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
              
              //const mfem::real_t fac = face_is_flipped ? -1.0 : 1.0;
              const mfem::real_t fac = 1.0;
              store(fslot, fp_restr, nor, fac/(w0*J1), fac/(w0*J2));
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
              
              //const mfem::real_t fac = face_is_flipped ? -1.0 : 1.0;
              const mfem::real_t fac1 = 1.0;
              const mfem::real_t fac2 = 0.0;
              store(fslot, fp_restr, nor, fac1/(w0*J1), fac2/(w0*J2));
            }
        } // Shared face processing
      } // Interior face processing
  }

  template<typename CacheT>
  void ComputeSubcellMetrics(mfem::FiniteElementSpace *fes, CacheT *cache)
  {
    auto *pfes = dynamic_cast<mfem::ParFiniteElementSpace *>(fes);
    auto *pmesh = pfes->GetMesh();
    const int dim = cache->dim;
    const int ne = pmesh->GetNE();
    const int Np_x = cache->Np_x;
    const int Np_y = cache->Np_y;
    const int Np_z = cache->Np_z;
    const mfem::real_t *D = cache->D.HostRead();
 
    mfem::DenseTensor SubcellMetricXi, SubcellMetricEta, SubcellMetricZeta;
    SubcellMetricXi.SetSize(dim, Np_z * Np_y * (Np_x + 1), ne);
    if(dim > 1)
      SubcellMetricEta.SetSize(dim, Np_z * (Np_y + 1) * Np_x, ne);
    if(dim == 3)
      SubcellMetricZeta.SetSize(dim, (Np_z + 1) * Np_y * Np_x, ne);
    
    for (int el = 0; el < ne; el++)
      {
        mfem::ElementTransformation *Tr = pmesh->GetElementTransformation(el);
        mfem::DenseMatrix &nor_mat_xi = SubcellMetricXi(el);

        mfem::Vector tmp(dim);
        mfem::Vector nor(dim);
        mfem::Vector metric1(dim);
        mfem::Vector metric2(dim);

        for (int k = 0; k < Np_z; k++)
          {
            const int pos1 = k * Np_y * Np_x;
            for (int j = 0; j < Np_y; j++)
              {
                const int pos = pos1 + j * Np_x;
                const mfem::IntegrationPoint &ip1 = cache->ir_vol->IntPoint(pos); // left xi-face
                Tr->SetIntPoint(&ip1);
                Tr->AdjugateJacobian().GetRow(0, metric1);
                for (int i = 0; i < Np_x + 1; i++)
                  {
                    nor = metric1;
                    for (int l = 0; l < i; l++)
                      {
                        const mfem::real_t *Dcol = D + l*Np_x; // D_T.GetColumn(l, D_row);
                        tmp = 0.0;
                        mfem::real_t weight = cache->ir->IntPoint(l).weight;
                        for (int m = 0; m < Np_x; m++)
                          {
                            const mfem::IntegrationPoint &ip2 = cache->ir_vol->IntPoint(pos + m);
                            Tr->SetIntPoint(&ip2);
                            Tr->AdjugateJacobian().GetRow(0, metric2);
                            metric2 *= Dcol[m];
                            tmp += metric2;
                          }
                        tmp *= weight;
                        nor += tmp;
                      }
                    nor_mat_xi.SetCol(pos + i, nor);
                  }
              }
          }
        
        if (dim > 1)
          {
            mfem::DenseMatrix &nor_mat_eta = SubcellMetricEta(el);
            for (int k = 0; k < Np_z; k++)
              {
                const int pos1 = k * Np_y * Np_x;
                for (int i = 0; i < Np_x; i++)
                  {
                    const mfem::IntegrationPoint &ip1 = cache->ir_vol->IntPoint(pos1 + i); // bottom eta-face
                    Tr->SetIntPoint(&ip1);
                    Tr->AdjugateJacobian().GetRow(1, metric1);
                    for (int j = 0; j < Np_y + 1; j++)
                      {
                        nor = metric1;
                        for (int l = 0; l < j; l++)
                          {
                            const mfem::real_t *Dcol = D + l*Np_x; // D_T.GetColumn(l, D_row);
                            tmp = 0.0;
                            mfem::real_t weight = cache->ir->IntPoint(l).weight;
                            for (int m = 0; m < Np_y; m++)
                              {
                                const int pos = pos1 + m * Np_x;
                                const mfem::IntegrationPoint &ip2 = cache->ir_vol->IntPoint(pos + i);
                                Tr->SetIntPoint(&ip2);
                                Tr->AdjugateJacobian().GetRow(1, metric2);
                                metric2 *= Dcol[m];
                                tmp += metric2;
                              }
                            tmp *= weight;
                            nor += tmp;
                          }
                        nor_mat_eta.SetCol(pos1 + j * Np_x + i, nor);
                      }
                  }
              }
            
            if (dim > 2)
              {
                mfem::DenseMatrix &nor_mat_zeta = SubcellMetricZeta(el);
                for (int j = 0; j < Np_y; j++)
                  {
                    const int pos1 = j * Np_x;
                    for (int i = 0; i < Np_x; i++)
                      {
                        const mfem::IntegrationPoint &ip1 = cache->ir_vol->IntPoint(pos1 + i); // bottom zeta-face
                        Tr->SetIntPoint(&ip1);
                        Tr->AdjugateJacobian().GetRow(2, metric1);
                        for (int k = 0; k < Np_z + 1; k++)
                          {
                            nor = metric1;
                            for (int l = 0; l < k; l++)
                              {
                                const mfem::real_t *Dcol = D + l*Np_x; // D_T.GetColumn(l, D_row);
                                tmp = 0.0;
                                mfem::real_t weight = cache->ir->IntPoint(l).weight;
                                for (int m = 0; m < Np_z; m++)
                                  {
                                    const int pos = m * Np_y * Np_x + pos1 + i;
                                    const mfem::IntegrationPoint &ip2 = cache->ir_vol->IntPoint(pos);
                                    Tr->SetIntPoint(&ip2);
                                    Tr->AdjugateJacobian().GetRow(2, metric2);
                                    metric2 *= Dcol[m];
                                    tmp += metric2;
                                  }
                                tmp *= weight;
                                nor += tmp;
                              }
                            nor_mat_zeta.SetCol(k * Np_y * Np_x + pos1 + i, nor);
                          }
                      }
                  }
              }
          }
      }
    CacheSubcellMetricData(SubcellMetricXi, SubcellMetricEta, SubcellMetricZeta, cache);
  }

  // Call this *after* element geometric cache is created
  template <typename CacheT>
  void CacheSubcellMetricData(const mfem::DenseTensor &SubcellMetricXi,
                              const mfem::DenseTensor &SubcellMetricEta,
                              const mfem::DenseTensor &SubcellMetricZeta,
                              CacheT *cache)
  {
    const int ne = cache->num_elements;
    const int dim = cache->dim;

    // Num quadrature points per direction
    const int Np_x = cache->Np_x;
    const int Np_y = cache->Np_y;
    const int Np_z = cache->Np_z;

    // Num points per metric component per element
    const int n_metric_xi = (Np_x + 1) * Np_y * Np_z;
    const int n_metric_eta = Np_x * (Np_y + 1) * Np_z;
    const int n_metric_zeta = Np_x * Np_y * (Np_z + 1);

    // Num metric values per element
    const int nq_metric_xi = n_metric_xi * dim;
    const int nq_metric_eta = n_metric_eta * dim;
    const int nq_metric_zeta = n_metric_zeta * dim;

    cache->subcellWeights.SetSize(Np_x);
    mfem::real_t *wgts = cache->subcellWeights.HostWrite();
    for (int i = 0;i < Np_x;i++){
      wgts[i] = cache->ir->IntPoint(i).weight;
    }

    cache->subcellMetricXi.SetSize(nq_metric_xi*ne);
    if (dim > 1)
      cache->subcellMetricEta.SetSize(nq_metric_eta*ne);
    if (dim > 2)
      cache->subcellMetricZeta.SetSize(nq_metric_zeta*ne);
    
    mfem::real_t *xi_h   = cache->subcellMetricXi.HostWrite();
    mfem::real_t *eta_h  = (dim > 1 ? cache->subcellMetricEta.HostWrite() : nullptr);
    mfem::real_t *zeta_h  = (dim > 2 ? cache->subcellMetricZeta.HostWrite() : nullptr);

    for (int el = 0; el < ne; ++el){
      const mfem::DenseMatrix &Mxi  = SubcellMetricXi(el);
      MFEM_VERIFY(Mxi.Height() == dim && Mxi.Width() == n_metric_xi,
                  "SubcellMetricXi has unexpected shape.");
      for (int id = 0; id < n_metric_xi; ++id){
        for (int d = 0; d < dim; ++d){
          xi_h [((el * n_metric_xi + id) * dim) + d] = Mxi(d, id);
        }
      }
      if (dim > 1) {
        const mfem::DenseMatrix &Meta = SubcellMetricEta(el);
        MFEM_VERIFY(Meta.Height() == dim && Meta.Width() == n_metric_eta,
                    "SubcellMetricEta has unexpected shape.");
        for (int id = 0; id < n_metric_eta; ++id)
          {
            for (int d = 0; d < dim; ++d)
              {
                eta_h[((el * n_metric_eta + id) * dim) + d] = Meta(d, id);
              }
          }
      }
      if (dim == 3)
        {
          const mfem::DenseMatrix &Mzeta = SubcellMetricZeta(el);
          MFEM_VERIFY(Mzeta.Height() == dim && Mzeta.Width() == n_metric_zeta,
                      "SubcellMetricZeta has unexpected shape.");
          for (int id = 0; id < n_metric_zeta; ++id)
            {
              for (int d = 0; d < dim; ++d)
                {
                  zeta_h[((el * n_metric_zeta + id) * dim) + d] = Mzeta(d, id);
                }
            }
        }
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

    // - Volume element data
    device_cache.elJac_d = cache.elJac.Read();
    device_cache.elMetric_d = cache.elMetric.Read();
    device_cache.D_d = cache.D.Read();
    device_cache.Dhat_d = cache.Dhat.Read();
    device_cache.Dhat2_d = cache.Dhat2.Read();
    device_cache.elQWgts_d = cache.elQuadratureWeights.Read();

    // - Interior faces (including remote)
    device_cache.nor_d = cache.face_normals.Read();
    device_cache.fw_minus_d = cache.face_wt_minus.Read();
    device_cache.fw_plus_d = cache.face_wt_plus.Read();

    // - Boundary faces
    device_cache.bnd_nor_d = cache.bnd_normals.Read();
    device_cache.bnd_wt_d = cache.bnd_wt.Read();
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
    MFEM_VERIFY(ds_size == cache.num_elements, "Element wavespeeds missized.");
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
