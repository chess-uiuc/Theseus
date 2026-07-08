// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause

namespace Theseus
{
  // This is a Inf/NAN detector helper
  int CBE(const mfem::Vector &v)
  {
    mfem::Vector one_bad(v.Size());
    const mfem::real_t *vd = v.Read();
    mfem::real_t *bd_w = one_bad.Write();

    mfem::forall(v.Size(), [=] MFEM_HOST_DEVICE (int i) {
      bd_w[i] = Theseus::Kernels::is_bad_value(vd[i]) ? 1.0 : 0.0;
    });
    const mfem::real_t *bd_r = one_bad.Read();
    int nbad = 0;
    for(int i = 0;i < v.Size();i++){
      nbad += bd_r[i];
    }
    return nbad;
  }

  // Assemble volume part of RHS for all elements
  // NOTE:
  //  - No axisymmetry (temporarily disabled in device version of MULT)
  template<typename PhysicsT>
  mfem::real_t EulerOperator<PhysicsT>::MultEuler_Volume(const mfem::Vector &pu, mfem::Vector &pdudt) const
  {
    Theseus::ScopedTimer timer("MultEuler_Volume");
    
    // This block is executed by the host
    int nval_restr = operator_cache.restr_v->Height();
    if(operator_cache.uVol.Size() != nval_restr){
      operator_cache.uVol.SetSize(nval_restr);
      operator_cache.uVol.UseDevice();
    }
    mfem::Vector &Ue(operator_cache.uVol);
    if(operator_cache.rhsVol.Size() != nval_restr){
      operator_cache.rhsVol.SetSize(nval_restr);
      operator_cache.rhsVol.UseDevice();
    }
    mfem::Vector &dUe(operator_cache.rhsVol);

    operator_cache.restr_v->Mult(pu, Ue);
  
    // Zero the array on-device
    {
      mfem::real_t *d = dUe.Write();
      mfem::forall(dUe.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = mfem::real_t(0); });
    }

    const mfem::real_t *Ue_d = Ue.Read();
    mfem::real_t *dUe_d = dUe.Write();

    // Copy the device cache so that it is not member data
    auto dc = device_cache;

    // Device cache parameters
    const int dim = dc.dim;
    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;

#ifdef SUBCELL_FV_BLENDING
    const int Np_x = dc.Np_x;
    const int Np_y = dc.Np_y;
    const int Np_z = dc.Np_z;
    const int npe = Np_x * Np_y * Np_z;
    const int ndofe = npe * neq;
    const int npe_metric_xi = (Np_x + 1)*Np_y*Np_z;
    const int npe_metric_eta = Np_x*(Np_y + 1)*Np_z;
    const int npe_metric_zeta = Np_x * Np_y * (Np_z + 1);
    const mfem::real_t *metric_xi_d = dc.subcell_metric_xi_d;
    const mfem::real_t *metric_eta_d = (dim > 1 ? dc.subcell_metric_eta_d : nullptr);
    const mfem::real_t *metric_zeta_d = (dim > 2 ? dc.subcell_metric_zeta_d : nullptr);

    mfem::Vector dUfv(operator_cache.restr_v->Height());
    dUfv.UseDevice();
    mfem::real_t *dUfv_d = dUfv.Write();
    // zero the array on-device
    {
      mfem::real_t *d = dUfv_d;
      mfem::forall(dUfv.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = mfem::real_t(0); });
    }

    const mfem::real_t *alpha_d = operator_cache.alpha->Read();
#endif

    // Derived parameters
    const int metric_stride = ndof * dim * dim;
    const int jac_stride    = ndof;
    const int estride = ndof*neq;
  
    // Device cache data/arrays
    const int *elem_attr_d = dc.elem_attr_d;
    const int *attr_marker_d = dc.attr_marker_d;
    const mfem::real_t *elJac_d = dc.elJac_d;
    const mfem::real_t *elMetric_d = dc.elMetric_d;

    mfem::real_t *ws_d = dc.elWaveSpeed_d;

    // Inside the FORALL below, executed on device
    mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
    {
    
      const mfem::real_t *jac_el    = elJac_d    + e * jac_stride;
      const mfem::real_t *metric_el = elMetric_d + e * metric_stride;

      const int attr = elem_attr_d[e];
      if (attr_marker_d[attr-1] == 0) {
        ws_d[e] = 0.0;
        return;
      }

      const int eoff = e * estride;
      const mfem::real_t *u_el = Ue_d + eoff;
      mfem::real_t *du_el = dUe_d + eoff;

      mfem::real_t cs_el = \
        DGSEMIntegrator::AssembleElementVolumeKernel(dc, u_el,
                                                     jac_el, metric_el, du_el);
#ifdef SUBCELL_FV_BLENDING
      mfem::real_t alpha_fv = alpha_d[e];
      if(alpha_fv > 1e-16){
        mfem::real_t alpha_inv = (1.0 - alpha_fv);
        mfem::real_t *du_fv = dUfv_d + eoff;
        const mfem::real_t *el_metric_xi = metric_xi_d + e * npe_metric_xi * dim;
        const mfem::real_t *el_metric_eta = (dim > 1 ? metric_eta_d + e * npe_metric_eta * dim :
                                             nullptr);
        const mfem::real_t *el_metric_zeta = (dim > 2 ? metric_zeta_d + e * npe_metric_zeta * dim :
                                              nullptr);
        const mfem::real_t cs_fv =                                              \
          DGSEMIntegrator::ComputeFVFluxesKernel(dc, u_el, jac_el, el_metric_xi, el_metric_eta, el_metric_zeta, du_fv);
      
        for(int ipt = 0;ipt < estride;ipt++){
          du_el[ipt] = alpha_inv * du_el[ipt] + alpha_fv * du_fv[ipt];
        }

        cs_el = Kernels::rmax(cs_el, cs_fv);
      }
#endif

      ws_d[e] = cs_el;

    });

    // Scatter RHS back to storage
    operator_cache.restr_v->AddMultTranspose(dUe, pdudt);

    // Finish up on the host:
    //  - Reduce for rank-local max_char_speed
    const mfem::real_t *ws = operator_cache.elWaveSpeed.HostRead();
    mfem::real_t max_char_speed = 0.0;
    for(int e = 0;e < operator_cache.num_elements;e++)
      {
        max_char_speed = std::max(max_char_speed, ws[e]);
      }

    return max_char_speed;
  }

  template<typename PhysicsT>
  mfem::real_t EulerOperator<PhysicsT>::MultEuler_InteriorFaces(const mfem::Vector &pu, mfem::Vector &pdudt) const
  {
    Theseus::ScopedTimer timer("MultEuler_InteriorFaces");
    auto dc = device_cache;
    const int dim = dc.dim;
    const int neq = dc.num_equations;
    const int nfp = dc.num_face_points;
    const int nval_restr = operator_cache.restr_f->Height();
    const int nfaces = nval_restr / (nfp * neq * 2); // (+/-)
    const int face_stride = 2 * nfp * neq;
    const int side_stride = nfp * neq;
    const int face_size = 2*nfp*neq;
    const int norm_size = nfp*dim;
  
    if(operator_cache.uInt.Size() != nval_restr){
      operator_cache.uInt.SetSize(nval_restr);
      operator_cache.uInt.UseDevice();
    }
    if(operator_cache.rhsInt.Size() != nval_restr){
      operator_cache.rhsInt.SetSize(nval_restr);
      operator_cache.rhsInt.UseDevice();
    }
    mfem::Vector &u_faces(operator_cache.uInt);
    mfem::Vector &rhs_faces(operator_cache.rhsInt);

    // For now, just keep this copy - i think it is device-friendly
    mfem::Vector faces_dudt(pdudt);
    faces_dudt.UseDevice();

    // If zeroed before accumulation, do it explicitly on device:
    // Potentially, this is not needed at all since I think we overwrite everything
    {
      mfem::real_t *d = rhs_faces.Write();
      mfem::forall(rhs_faces.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = mfem::real_t(0); });
    }

    operator_cache.restr_f->Mult(pu, u_faces);

    const mfem::real_t *u_d = u_faces.Read();
    mfem::real_t *rhs_d = rhs_faces.Write();

    const mfem::real_t *nor_d   = dc.nor_d;      // size nfaces*nfp*dim
    const mfem::real_t *inv1_d  = dc.fw_minus_d; // size nfaces*nfp
    const mfem::real_t *inv2_d  = dc.fw_plus_d;  // size nfaces*nfp

    mfem::real_t *ws_d = dc.ifWaveSpeed_d;

    mfem::forall(nfaces, [=] MFEM_HOST_DEVICE (int i)
    {
      const int face_offset = i*face_size;
      const int n_offset = i*norm_size;
      const int w_offset = i*nfp;

      const mfem::real_t *u_face_d = u_d + face_offset;
      mfem::real_t *rhs_face_d = rhs_d + face_offset;
      const mfem::real_t *nor_face_d = nor_d + n_offset;
      const mfem::real_t *w_minus_d = inv1_d + w_offset;
      const mfem::real_t *w_plus_d = inv2_d + w_offset;
    
      mfem::real_t ws = DGSEMIntegrator::AssembleElementFaceKernel(dc, u_face_d, nor_face_d,
                                                                   w_minus_d, w_plus_d, rhs_face_d);
      ws_d[i] = ws;
    
    });

    operator_cache.restr_f->MultTranspose(rhs_faces, faces_dudt);
    pdudt += faces_dudt; // on device? 

    // Finish up on the host:
    //  - Reduce for rank-local max_char_speed
    const mfem::real_t *ws = operator_cache.ifWaveSpeed.HostRead();
    mfem::real_t max_char_speed_facial = 0.0;
    for(int f = 0;f < operator_cache.num_interior_faces;f++)
      {
        max_char_speed_facial = std::max(max_char_speed_facial, ws[f]);
      }

    return max_char_speed_facial;
  }

  template<typename PhysicsT>
  mfem::real_t EulerOperator<PhysicsT>::MultEuler_BoundaryFaces(const mfem::Vector &pu, mfem::Vector &pdudt) const
  {
    Theseus::ScopedTimer timer("MultEuler_BoundaryFaces");

    auto dc = device_cache;
    const int dim = dc.dim;
    const int neq = dc.num_equations;
    const int nfp = dc.num_face_points;
    const int face_size = nfp * neq;
    const int restr_size = operator_cache.restr_b->Height();
    const int nfaces_restr = restr_size / face_size;
    const int norm_size = nfp * dc.dim;
    const int npoints_bnd = nfaces_restr * nfp;

    if(restr_size == 0){
      return 0.0;
    }

    if(operator_cache.uBnd.Size() != restr_size)
      {
        operator_cache.uBnd.SetSize(restr_size);
        operator_cache.uBnd.UseDevice();
      }
    if(operator_cache.rhsBnd.Size() != restr_size){
      operator_cache.rhsBnd.SetSize(restr_size);
      operator_cache.rhsBnd.UseDevice();
    }
    if(operator_cache.dudtBnd.Size() != restr_size){
      operator_cache.dudtBnd.SetSize(pdudt.Size());
      operator_cache.dudtBnd.UseDevice();
    }

    mfem::Vector &u_faces(operator_cache.uBnd);
    mfem::Vector &rhs_faces(operator_cache.rhsBnd);
    mfem::Vector faces_dudt(pdudt);

    // If zeroed before accumulation, do it explicitly on device:
    // Potentially, this is not needed at all since I think we overwrite everything
    {
      mfem::real_t *rd = rhs_faces.Write();
      mfem::forall(rhs_faces.Size(), [=] MFEM_HOST_DEVICE (int i)
      { rd[i] = mfem::real_t(0);});
      mfem::real_t *fd = faces_dudt.Write();
      mfem::forall(faces_dudt.Size(), [=] MFEM_HOST_DEVICE (int i)
      { fd[i] = mfem::real_t(0);});
    }

    operator_cache.restr_b->Mult(pu, u_faces);

    const mfem::real_t *u_d = u_faces.Read();
    mfem::real_t *rhs_d = rhs_faces.Write();

    const mfem::real_t *nor_d   = dc.bnd_nor_d;      // size nfaces*nfp*dim
    const mfem::real_t *inv1_d  = dc.bnd_wt_d; // size nfaces*nfp
    const int *bnd_marker_index_d = dc.bnd_marker_index_d;
    mfem::real_t *ws_d = dc.bndWaveSpeed_d;

    mfem::forall(npoints_bnd, [=] MFEM_HOST_DEVICE (int p)
    {
      const int f = p / nfp;
      const int fp = p % nfp;

      int bnd_face_marker_index = bnd_marker_index_d[f];
      if(bnd_face_marker_index < 0){
        ws_d[p] = 0.0;
        return;
      }
      //    int bc_index = bnd_marker_to_bc_descr_d[bnd_face_marker_index];
      int bc_index = bnd_face_marker_index; // no mapping atm
      if(bc_index < 0){
        ws_d[p] = 0.0;
        return;
      }
      const Theseus::BCDescriptor &bc = dc.bc_descr_d[bc_index];
      if (bc.type == int(Theseus::BCType::Invalid))
        {
          ws_d[p] = 0.0;
          return;
        }

      const int face_offset = f * face_size;
      const int n_offset = f * norm_size;
      const int w_offset = f * nfp;

      const mfem::real_t *u_face_d = u_d + face_offset;
      mfem::real_t *rhs_face_d = rhs_d + face_offset;
      const mfem::real_t *nor_face_d = nor_d + n_offset;
      const mfem::real_t *w_minus_d = inv1_d + w_offset;
      const mfem::real_t *nor_point = nor_face_d + fp*dim;
      mfem::real_t scale = -w_minus_d[fp];
      // #ifdef AXISYMMETRIC
      // NOTE: axisymmetric not ready for device yet
      // scale *= rad_face[fp];
      // #else
      // #error "Axisymmetric boundary device path not implemented yet."
      // #endif
      mfem::real_t state1[Theseus::MAXEQ];
      mfem::real_t fluxN[Theseus::MAXEQ];

      Theseus::Kernels::el_gather_state(u_face_d, nfp, neq, fp, state1);
      const mfem::real_t ws = \
        Theseus::BC::ApplyBoundaryConditionInviscid(dc, bc, state1,
                                                    nor_point, fluxN);
      Theseus::Kernels::el_scatter_add(fluxN, nfp, neq, fp, scale, rhs_face_d);
      ws_d[p] = ws;

    });

    operator_cache.restr_b->MultTranspose(rhs_faces, faces_dudt);

    pdudt += faces_dudt; // on device? (likely yes) 

    // Finish up on the host:
    //  - Reduce for rank-local max_char_speed
    const mfem::real_t *ws = operator_cache.bndWaveSpeed.HostRead();
    mfem::real_t max_char_speed_facial = 0.0;
    for(int p = 0;p < npoints_bnd;p++)
      {
        max_char_speed_facial = std::max(max_char_speed_facial, ws[p]);
      }

    return max_char_speed_facial;
  }


  // Top level MULT for inviscid cases, called from DGSEMOperator
  template<typename PhysicsT>
  mfem::real_t EulerOperator<PhysicsT>::FlowMult(const mfem::Vector &u, mfem::Vector &dudt) const
  {
    Theseus::ScopedTimer timer("EulerMult");
    
    auto report_bad = [&](const char *name, const mfem::Vector &v)
    {
      int nbad = CBE(v);
      if (nbad)
        {
          mfem::out << "BAD VALUES IN: (" << name << "), count=" << nbad << std::endl;
        }
    };

    // Theseus::ScopedTimer timer("MultInviscid");
    const mfem::Vector &pu = this->Prolongate(u);
    if (this->P)
      {
        operator_cache.pdudt.SetSize(this->P->Height());
      }

    mfem::Vector &pdudt = this->P ? operator_cache.pdudt : dudt;
    int psize = pdudt.Size();

    // Zero on-device
    mfem::real_t *pdudt_d = pdudt.Write();
    mfem::forall(psize, [=] MFEM_HOST_DEVICE (int i) { pdudt_d[i] = 0.0; });

    mfem::real_t max_char_speed = 0.0;
    // This step overwrites contents of pdudt
    max_char_speed = MultEuler_Volume(pu, pdudt);
    mfem::real_t max_char_speed_facial = 0.0;
    max_char_speed_facial = MultEuler_InteriorFaces(pu, pdudt);
    // report_bad("int rhs", pdudt);

    max_char_speed = std::max(max_char_speed, max_char_speed_facial);
    mfem::real_t max_char_speed_bnd = 0.0;
    max_char_speed_bnd = MultEuler_BoundaryFaces(pu, pdudt);
    // report_bad("bnd rhs", pdudt);

    max_char_speed = std::max(max_char_speed, max_char_speed_bnd);

    if (this->Serial())
      {
        if(this->cP) this->cP->MultTranspose(pdudt, dudt);
      }
    else
      {
        if(this->P) this->P->MultTranspose(pdudt, dudt);
      }

    const int N = this->ess_tdof_list.Size();
    const auto idx = this->ess_tdof_list.Read();
    auto DU_RW = dudt.ReadWrite();
    mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { DU_RW[idx[i]] = 0.0; });

    return max_char_speed;
  }
  
}
