// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause

namespace Theseus
{

  // Device version of ComputeGlobalEntropyVector
  template<typename PhysicsT>
  void NSOperator<PhysicsT>::ComputeEntropyState(const mfem::Vector &u, mfem::Vector &e) const
  {
    Theseus::ScopedTimer timer("ComputeEntropyState");

    // This block is executed by the host
    const int nval_restr = operator_cache.restr_v->Height();

    // Copy the device cache so that it is not member data
    auto dc = device_cache;

    // Device cache parameters
    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;
    const int npts = ndof * ne;

    MFEM_ASSERT(nval_restr == npts*neq, "Unexpected size in ComputeEntropyState");

    auto gas = dc.gas;

    if (operator_cache.uVol.Size() != nval_restr){
      operator_cache.uVol.SetSize(nval_restr);
      operator_cache.uVol.UseDevice();
    }
    mfem::Vector &restrU(operator_cache.uVol);
    if (operator_cache.uVol.Size() != nval_restr){
      operator_cache.uVol.SetSize(nval_restr);
      operator_cache.uVol.UseDevice();
    }
    mfem::Vector &restrE(operator_cache.uVol);
    if(e.Size() != u.Size()){
      e.SetSize(u.Size());
      e.UseDevice();
    }

    mfem::real_t *eState_d = restrE.Write();
    operator_cache.restr_v->Mult(u, restrU);
    const mfem::real_t *restrU_d = restrU.Read();
    const int estride = ndof*neq;

    // Inside the FORALL below, executed on device
    mfem::forall(npts, [=] MFEM_HOST_DEVICE (int pt)
    {
      const int elno = pt / ndof;
      const int ept = pt % ndof;
      const int eoff = elno * estride;
      const mfem::real_t *u_el = restrU_d + eoff;

      mfem::real_t elUstate[Theseus::MAXEQ];
      Theseus::Kernels::el_gather_state(u_el, ndof, neq, ept, elUstate);
      Theseus::PointStateView S{elUstate};

      mfem::real_t elEState[Theseus::MAXEQ];
      Theseus::PointStateViewRW E{elEState};
      gas.entropy_state(S, E);
      mfem::real_t *e_el = eState_d + eoff;
      Theseus::Kernels::el_scatter_assign(elEState, ndof, neq, ept, 1.0, e_el);
    });

    operator_cache.restr_v->MultTranspose(restrE, e);

  }

  // This routine replaces the gradient of the entropy stored in gradState with the gradient
  // of the primitive variables so that all the data in gradState is replaced.
  template<typename PhysicsT>
  void NSOperator<PhysicsT>::ComputeGradPrimFromGradEntropy(const mfem::Vector &u, std::vector<mfem::Vector *> &gradEntropy) const
  {
    Theseus::ScopedTimer timer("GradEntropyToGradPrim");
    // This block is executed by the host
    const int nval_restr = operator_cache.restr_v->Height();

    // Copy the device cache so that it is not member data
    auto dc = device_cache;

    // Device cache parameters
    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;
    const int npts = ndof * ne;
    const int dim = dc.dim;

    MFEM_ASSERT(nval_restr == npts*neq, "Unexpected size in ComputeEntropyState");

    auto gas = dc.gas;

    if (operator_cache.uVol.Size() != nval_restr){
      operator_cache.uVol.SetSize(nval_restr);
      operator_cache.uVol.UseDevice();
    }
    mfem::Vector &restr_state(operator_cache.uVol);
    operator_cache.restr_v->Mult(u, restr_state);

    const mfem::real_t *restr_state_d = restr_state.Read();
    const int estride = ndof*neq;

    // Leave this temporary for now
    if(operator_cache.volAux.Size() != nval_restr){
      operator_cache.volAux.SetSize(nval_restr);
      operator_cache.volAux.UseDevice();
    }
    mfem::Vector &restr_grad_prim_dir(operator_cache.volAux);

    for(int idim = 0;idim < dim;idim++){

      mfem::Vector &grad_state_dir(*gradEntropy[idim]);
      operator_cache.restr_v->Mult(grad_state_dir, restr_grad_prim_dir);
      mfem::real_t *grad_prim_dir_d = restr_grad_prim_dir.Write();

      // Inside the FORALL below, executed on device
      mfem::forall(npts, [=] MFEM_HOST_DEVICE (int pt)
      {
        const int e = pt / ndof;
        const int ept = pt % ndof;
        const int eoff = e * estride;
        const mfem::real_t *u_el = restr_state_d + eoff;
        mfem::real_t *grad_prim_el = grad_prim_dir_d + eoff;

        mfem::real_t el_U[Theseus::MAXEQ];
        Theseus::Kernels::el_gather_state(u_el, ndof, neq, ept, el_U);
        Theseus::PointStateView CV{el_U};

        mfem::real_t el_gradS[Theseus::MAXEQ];
        Theseus::Kernels::el_gather_state(grad_prim_el, ndof, neq, ept, el_gradS);
        Theseus::PointStateView dS{el_gradS};

        mfem::real_t el_gradP[Theseus::MAXEQ];
        Theseus::PointStateViewRW dP{el_gradP};
        gas.grad_entropy_to_grad_prim(CV, dS, dP);

        Kernels::el_scatter_assign(el_gradP, ndof, neq, ept, 1.0, grad_prim_el);

      });

      operator_cache.restr_v->MultTranspose(restr_grad_prim_dir, grad_state_dir);

    }
  }

  template<typename PhysicsT>
  mfem::real_t NSOperator<PhysicsT>::FlowMult(const mfem::Vector &u, mfem::Vector &dudt) const
  {
    Theseus::ScopedTimer timer("NSRHS");

    const mfem::Vector &Ustate = u;
    const int dim = operator_cache.dim;
    {
      Theseus::ScopedTimer timer_step("Step");
      mfem::Vector &entropyState(operator_cache.entropyState);
      {
        Theseus::ScopedTimer etime("EntropyPlumbing");
        if (entropyState.Size() != Ustate.Size()){
          entropyState.SetSize(Ustate.Size());
          entropyState.UseDevice();
        }
        ComputeEntropyState(Ustate, entropyState);
      }
      std::vector<mfem::Vector *> gradPrim(dim);
      {
        Theseus::ScopedTimer gtime("GradientPlumbing");
        // grad_u is a vector of parallel grid functions
        // this bit grabs an mfem::Vector ref.
        // Note that incoming grad_u is really grad of entropy,
        // which we pack into gradPrim, and then call a
        // function which overwrites the entropy gradient
        // with the primitive gradient.
        for(int idim = 0;idim < dim;idim++){
          gradPrim[idim] = &(*grad_u[idim]);
        }
        GradOperator(entropyState, gradPrim);
        ComputeGradPrimFromGradEntropy(Ustate, gradPrim);
      }
      max_char_speed = MultCNS(Ustate, gradPrim, dudt);
    }
    return max_char_speed;
  }

  template<typename PhysicsT>
  void NSOperator<PhysicsT>::GradOperator_Volume(const mfem::Vector &pu,
                                                 std::vector<mfem::Vector *> &p_grad_u) const
  {
    Theseus::ScopedTimer timer("GradOperator_Volume");

    const int dim = operator_cache.dim;
    const int restr_size = operator_cache.restr_v->Height();

    if(operator_cache.uVol.Size() != restr_size){
      operator_cache.uVol.SetSize(restr_size);
      operator_cache.uVol.UseDevice();
    }
    mfem::Vector &Ue(operator_cache.uVol);
    mfem::real_t *dU_d[Theseus::MAXDIM] = {nullptr, nullptr, nullptr};
    mfem::real_t *pgrad_d[Theseus::MAXDIM] = {nullptr, nullptr, nullptr};
    if (operator_cache.gradVol.size() != dim){
      operator_cache.gradVol.resize(dim);
      for(int idim = 0;idim < dim;idim++){
        operator_cache.gradVol[idim].SetSize(restr_size);
        operator_cache.gradVol[idim].UseDevice();
      }
    }
    std::vector<mfem::Vector> &dUe(operator_cache.gradVol);
    for(int idim = 0;idim < dim;idim++){
      dU_d[idim] = dUe[idim].Write();
      pgrad_d[idim] = p_grad_u[idim]->Write();
    }

    mfem::forall(restr_size, [=] MFEM_HOST_DEVICE (int i)
    {
      for(int idim = 0;idim < dim;idim++){
        dU_d[idim][i] = mfem::real_t(0);
        pgrad_d[idim][i] = mfem::real_t(0);
      }
    });

    operator_cache.restr_v->Mult(pu, Ue);
    const mfem::real_t *Ue_d = Ue.Read();

    auto dc = device_cache;

    const int ne = dc.num_elements;
    const int ndof = dc.ndof_scalar_el;
    const int neq = dc.num_equations;
    const int estride = ndof * neq;
    const int jac_stride = ndof;
    const int metric_stride = ndof * dc.dim * dc.dim;

    const mfem::real_t *elJac_d = dc.elJac_d;
    const mfem::real_t *elMetric_d = dc.elMetric_d;

    mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e)
    {
      const mfem::real_t *u_el = Ue_d + e * estride;
      mfem::real_t *du_el_d[Theseus::MAXDIM] = {nullptr, nullptr, nullptr};
      for(int idim = 0;idim < dim;idim++){
        du_el_d[idim] = dU_d[idim] + e*estride;
      }

      const mfem::real_t *jac_el = elJac_d + e * jac_stride;
      const mfem::real_t *metric_el = elMetric_d + e * metric_stride;

      Theseus::DGSEMIntegrator::AssembleGradElementVolumeKernel(dc, u_el, jac_el, metric_el,
                                                                du_el_d);
    });

    for(int idim = 0;idim < dim;idim++){
      operator_cache.restr_v->AddMultTranspose(dUe[idim], *p_grad_u[idim]);
    }

  }

  template<typename PhysicsT>
  void NSOperator<PhysicsT>::GradOperator_BoundaryFaces(const mfem::Vector &pu,
                                                        std::vector<mfem::Vector *> &p_grad_u) const
  {
    Theseus::ScopedTimer timer("GradOperator_BoundaryFaces");

    auto dc = device_cache;
    const int dim = dc.dim;
    const int neq = dc.num_equations;
    const int nfp = dc.num_face_points;
    const int face_size = nfp * neq;
    const int restr_size = operator_cache.restr_b->Height();
    const int nfaces_restr = restr_size / face_size;
    const int norm_size = nfp * dim;
    const int npoints_bnd = nfaces_restr * nfp;
    const int psize = pu.Size();

    if(restr_size == 0){
      return;
    }

    mfem::Vector &u_faces(operator_cache.uBnd);
    if(u_faces.Size() != restr_size){
      u_faces.SetSize(restr_size);
      u_faces.UseDevice();
    }

    std::vector<mfem::Vector> &rhs_faces(operator_cache.gradBnd);
    if(rhs_faces.size() != dim){
      rhs_faces.resize(dim);
      for(int idim = 0;idim < dim;idim++){
        rhs_faces[idim].SetSize(restr_size);
        rhs_faces[idim].UseDevice();
      }
    }

    mfem::Vector &duBnd(operator_cache.duBnd);
    if(duBnd.Size() != psize){
      duBnd.SetSize(psize);
      duBnd.UseDevice();
    }
    
    operator_cache.restr_b->Mult(pu, u_faces);
    
    const mfem::real_t *u_d = u_faces.Read();
    const mfem::real_t *nor_d = dc.bnd_nor_d;
    const mfem::real_t *wt_d = dc.bnd_wt_d;
    const int *bnd_marker_index_d = dc.bnd_marker_index_d;

    mfem::real_t *rhs_d[Theseus::MAXDIM] = {nullptr, nullptr, nullptr};
    mfem::real_t *du_d[Theseus::MAXDIM] = {nullptr, nullptr, nullptr};
    for(int idim = 0;idim < dim;idim++){
      rhs_d[idim] = rhs_faces[idim].Write();
      du_d[idim] = duBnd.Write();
    }

    for (int idim = 0; idim < dim; ++idim) {
      mfem::real_t *rd = rhs_d[idim];
      mfem::forall(restr_size, [=] MFEM_HOST_DEVICE (int i) { rd[i] = mfem::real_t(0); });
    }
    for (int idim = 0; idim < 1; ++idim) {
      mfem::real_t *dud = du_d[idim];
      mfem::forall(psize, [=] MFEM_HOST_DEVICE (int i) { dud[i] = mfem::real_t(0); });
    }

    mfem::forall(npoints_bnd, [=] MFEM_HOST_DEVICE (int p)
    {
      const int f = p / nfp;
      const int fp = p % nfp;

      const int bnd_face_marker_index = bnd_marker_index_d[f];
      if (bnd_face_marker_index < 0)
        {
          return;
        }

      const int bc_index = bnd_face_marker_index; // same convention as inviscid device path for now
      if (bc_index < 0)
        {
          return;
        }

      const Theseus::BCDescriptor &bc = dc.bc_descr_d[bc_index];
      if (bc.type == int(Theseus::BCType::Invalid))
        {
          return;
        }

      const int face_offset = f * face_size;
      const int norm_offset = f * norm_size;
      const int w_offset = f * nfp;

      const mfem::real_t *u_face_d = u_d + face_offset;

      const mfem::real_t *nor_face_d = nor_d + norm_offset;
      const mfem::real_t *nor_point = nor_face_d + fp * dim;

      // Legacy one-sided boundary lifting uses +1/(w0*J1)
      const mfem::real_t scale = wt_d[w_offset + fp];

      mfem::real_t *rhs_face[Theseus::MAXDIM] = {nullptr, nullptr, nullptr};
      for(int idim = 0;idim < dim;idim++){
        rhs_face[idim] = rhs_d[idim] + face_offset;
      }

      Theseus::DGSEMIntegrator::AssembleGradBoundaryPointKernel(dc, bc,
                                                                u_face_d,
                                                                nor_point,
                                                                scale,
                                                                fp,
                                                                rhs_face);
    });

    for(int idim = 0;idim < dim;idim++){
      operator_cache.restr_b->MultTranspose(rhs_faces[idim], duBnd);
      *p_grad_u[idim] += duBnd;
    }

  }

  template<typename PhysicsT>
  void NSOperator<PhysicsT>::GradOperator_InteriorFaces(const mfem::Vector &pu,
                                                        std::vector<mfem::Vector *> &p_grad_u) const
  {
    Theseus::ScopedTimer timer("GradOperator_InteriorFaces");

    auto dc = device_cache;
    const int dim = dc.dim;
    const int psize = pu.Size();
    const int restr_size = operator_cache.restr_f->Height();
    const int neq = dc.num_equations;
    const int nfp = dc.num_face_points;
    const int nfaces = restr_size / (2 * nfp * neq);
    const int face_size = 2 * nfp * neq;
    const int norm_size = nfp * dim;

    mfem::Vector &u_faces(operator_cache.uInt);
    if(u_faces.Size() != restr_size){
      u_faces.SetSize(restr_size);
      u_faces.UseDevice();
    }

    std::vector<mfem::Vector> &rhs_faces(operator_cache.gradInt);
    if(rhs_faces.size() != dim){
      rhs_faces.resize(dim);
      for(int idim = 0;idim < dim;idim++){
        rhs_faces[idim].SetSize(restr_size);
        rhs_faces[idim].UseDevice();
      }
    }

    mfem::Vector &duInt(operator_cache.duInt);
    if(duInt.Size() != psize){
      duInt.SetSize(psize);
      duInt.UseDevice();
    }

    operator_cache.restr_f->Mult(pu, u_faces);
    const mfem::real_t *u_d = u_faces.Read();

    mfem::real_t *rhs_d[Theseus::MAXDIM] = {nullptr, nullptr, nullptr};
    mfem::real_t *du_d[Theseus::MAXDIM] = {nullptr, nullptr, nullptr};
    for(int idim = 0;idim < dim;idim++){
      rhs_d[idim] = rhs_faces[idim].Write();
      // We only need 1dim at a time, lets only use 1 temp
      du_d[idim] = duInt.Write();
    }

    for (int idim = 0; idim < dim; ++idim) {
      mfem::real_t *rd = rhs_d[idim];
      mfem::forall(restr_size, [=] MFEM_HOST_DEVICE (int i) { rd[i] = mfem::real_t(0); });
    }
    // HardCode to dim=1 for now - temporary is only 1d
    for (int idim = 0; idim < 1; ++idim) {
      mfem::real_t *dud = du_d[idim];
      mfem::forall(psize, [=] MFEM_HOST_DEVICE (int i) { dud[i] = mfem::real_t(0); });
    }

    const mfem::real_t *nor_d  = dc.nor_d;
    const mfem::real_t *wm_d   = dc.fw_minus_d;
    const mfem::real_t *wp_d   = dc.fw_plus_d;

    mfem::forall(nfaces, [=] MFEM_HOST_DEVICE (int f)
    {
      const int face_offset = f * face_size;
      const int norm_offset = f * norm_size;
      const int w_offset    = f * nfp;

      const mfem::real_t *u_face_d    = u_d + face_offset;
      const mfem::real_t *nor_face_d  = nor_d + norm_offset;
      const mfem::real_t *w_minus_d   = wm_d + w_offset;
      const mfem::real_t *w_plus_d    = wp_d + w_offset;

      mfem::real_t *rhs_face[Theseus::MAXDIM] = {nullptr, nullptr, nullptr};
      for(int idim = 0;idim < dim;idim++){
        rhs_face[idim] = rhs_d[idim] + face_offset;
      }

      Theseus::DGSEMIntegrator::AssembleGradInteriorFaceKernel(dc,
                                                               u_face_d,
                                                               nor_face_d,
                                                               w_minus_d,
                                                               w_plus_d,
                                                               rhs_face);
    });

    for(int idim = 0;idim < dim;idim++){
      operator_cache.restr_f->MultTranspose(rhs_faces[idim], duInt);
      *p_grad_u[idim] += duInt;
    }

  }

  template<typename PhysicsT>
  void NSOperator<PhysicsT>::GradOperator(const mfem::Vector &u,
                                          std::vector<mfem::Vector *> &grad_u) const
  {
    Theseus::ScopedTimer timer("GradOperator");
    const int dim = operator_cache.dim;
    const mfem::Vector &pu = this->Prolongate(u);
    std::vector<mfem::Vector *> p_grad_(dim);
    if (this->P)
      {
        const int psize = this->P->Height();
        if(operator_cache.pGrad.size() != dim){
          operator_cache.pGrad.resize(dim);
          for(int idim = 0;idim < dim;idim++){
            operator_cache.pGrad[idim].SetSize(psize);
            operator_cache.pGrad[idim].UseDevice();
          }
        }
        for(int idim = 0;idim < dim;idim++){
          p_grad_[idim] = &(operator_cache.pGrad[idim]);
        }
      }
    std::vector<mfem::Vector *> &p_grad_u = this->P ? p_grad_ : grad_u;

    MFEM_ASSERT(p_grad_u.size() == dim, "Size mismatch for gradient storage");
    MFEM_ASSERT(grad_u.size() == dim, "Size mismatch for gradient storage");

    GradOperator_Volume(pu, p_grad_u);

    GradOperator_InteriorFaces(pu, p_grad_u);

    GradOperator_BoundaryFaces(pu, p_grad_u);

    if (this->Serial())
      {
        if (this->cP)
          {
            for(int idim = 0;idim < dim;idim++){
              this->cP->MultTranspose(*p_grad_u[idim], *grad_u[idim]);
            }
          }
      }
    else
      {
        if(this->P){
          for(int idim = 0;idim < dim;idim++){
            this->P->MultTranspose(*p_grad_u[idim], *grad_u[idim]);
          }
        }
      }

    const int N = this->ess_tdof_list.Size();
    const auto idx = this->ess_tdof_list.Read();
    for(int idim = 0;idim < dim;idim++){
      auto gradu_dim_d = grad_u[idim]->ReadWrite();
      mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { gradu_dim_d[idx[i]] = 0.0; });
    }
  }

  template<typename PhysicsT>
  mfem::real_t NSOperator<PhysicsT>::MultCNS_InteriorFaces(const mfem::Vector &pu,
                                                           const std::vector<mfem::Vector *> &p_grad_prim,
                                                           mfem::Vector &pdudt) const
  {
    Theseus::ScopedTimer timer("MultCNS_InteriorFaces");
  
    auto dc = device_cache;
    const int dim = dc.dim;
    const int neq = dc.num_equations;
    const int nfp = dc.num_face_points;
    const int nfaces = operator_cache.restr_f->Height() / (nfp * neq * 2); // (+/-)
    const int face_stride = 2 * nfp * neq;
    const int side_stride = nfp * neq;
    const int face_size = 2*nfp*neq;
    const int norm_size = nfp*dim;

    const int restr_size = operator_cache.restr_f->Height();
    mfem::Vector &int_u(operator_cache.uInt);
    if(int_u.Size() != restr_size){
      int_u.SetSize(restr_size);
      int_u.UseDevice();
    }

    mfem::Vector &rhs_faces(operator_cache.rhsInt);
    if(rhs_faces.Size() != restr_size){
      rhs_faces.SetSize(restr_size);
      rhs_faces.UseDevice();
    }

    mfem::Vector &faces_dudt(operator_cache.dudtInt);
    if(faces_dudt.Size() != pdudt.Size()){
      faces_dudt.SetSize(pdudt.Size());
      faces_dudt.UseDevice();
    }
    // faces_dudt = pdudt;

    const mfem::real_t *grad_prim_d[Theseus::MAXDIM] = {nullptr, nullptr, nullptr};
    std::vector<mfem::Vector> &int_grad_prim(operator_cache.gradInt);

    if(int_grad_prim.size() != dim){
      int_grad_prim.resize(dim);
      for(int idim = 0;idim < dim;idim++){
        int_grad_prim[idim].SetSize(restr_size);
        int_grad_prim[idim].UseDevice();
      }
    }

    for(int idim = 0;idim < dim;idim++){
      operator_cache.restr_f->Mult(*p_grad_prim[idim], int_grad_prim[idim]);
      grad_prim_d[idim] = int_grad_prim[idim].Read();
    }
  
    // If zeroed before accumulation, do it explicitly on device:
    // Potentially, this is not needed at all since I think we overwrite everything
    {
      mfem::real_t *d = rhs_faces.Write();
      mfem::forall(rhs_faces.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = mfem::real_t(0); });
    }

    operator_cache.restr_f->Mult(pu, int_u);

    const mfem::real_t *u_d = int_u.Read();
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
      const mfem::real_t *dprim_face_x = (dim > 0) ? grad_prim_d[0] + face_offset : nullptr;
      const mfem::real_t *dprim_face_y = (dim > 1) ? grad_prim_d[1] + face_offset : nullptr;
      const mfem::real_t *dprim_face_z = (dim > 2) ? grad_prim_d[2] + face_offset : nullptr;

      // Call one fused kernel for inviscid and viscous facial terms
      mfem::real_t ws = Theseus::DGSEMIntegrator::AssembleViscousElementFaceKernel(dc, u_face_d, nor_face_d,
                                                                                   w_minus_d, w_plus_d,
                                                                                   dprim_face_x,
                                                                                   dprim_face_y,
                                                                                   dprim_face_z,
                                                                                   rhs_face_d);
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
  mfem::real_t NSOperator<PhysicsT>::MultCNS_BoundaryFaces(const mfem::Vector &pu,
                                                           const std::vector<mfem::Vector *> &p_grad_prim,
                                                           mfem::Vector &pdudt) const
  {
    Theseus::ScopedTimer timer("MultCNS_BoundaryFaces");

    auto dc = device_cache;
    const int dim = dc.dim;
    const int neq = dc.num_equations;
    const int nfp = dc.num_face_points;
    const int face_size = nfp * neq;
    const int restr_size = operator_cache.restr_b->Height();
    const int nfaces_restr = restr_size / face_size;
    const int norm_size = nfp * dc.dim;
    const int npoints_bnd = nfaces_restr * nfp;
    const int psize = pdudt.Size();

    if(restr_size == 0){
      return 0.0;
    }

    mfem::Vector &rhs_faces(operator_cache.rhsBnd);
    mfem::Vector &faces_dudt(operator_cache.dudtBnd);
    if(rhs_faces.Size() != restr_size){
      rhs_faces.SetSize(restr_size);
      rhs_faces.UseDevice();
    }
    if(faces_dudt.Size() != psize){
      faces_dudt.SetSize(psize);
      faces_dudt.UseDevice();
    }
    mfem::Vector &bnd_u(operator_cache.uBnd);
    if(bnd_u.Size() != restr_size){
      bnd_u.SetSize(restr_size);
      bnd_u.UseDevice();
    }
    const mfem::real_t *grad_prim_d[Theseus::MAXDIM] = {nullptr, nullptr, nullptr};
    std::vector<mfem::Vector> &bnd_grad_prim(operator_cache.gradBnd);
    if(bnd_grad_prim.size() != dim){
      bnd_grad_prim.resize(dim);
      for(int idim = 0;idim < dim;idim++){
        bnd_grad_prim[idim].SetSize(restr_size);
        bnd_grad_prim[idim].UseDevice();
      }
    }
    for(int idim = 0;idim < dim;idim++){
      operator_cache.restr_b->Mult(*p_grad_prim[idim], bnd_grad_prim[idim]);
      grad_prim_d[idim] = bnd_grad_prim[idim].Read();
    }

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

    operator_cache.restr_b->Mult(pu, bnd_u);

    const mfem::real_t *u_d = bnd_u.Read();
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
      mfem::real_t gradPrim_x[Theseus::MAXEQ];
      mfem::real_t gradPrim_y[Theseus::MAXEQ];
      mfem::real_t gradPrim_z[Theseus::MAXEQ];
      const mfem::real_t *dprim_face_x = (dim > 0) ? grad_prim_d[0] + face_offset : nullptr;
      const mfem::real_t *dprim_face_y = (dim > 1) ? grad_prim_d[1] + face_offset : nullptr;
      const mfem::real_t *dprim_face_z = (dim > 2) ? grad_prim_d[2] + face_offset : nullptr;
      Theseus::Kernels::el_gather_grad_state(dprim_face_x, dprim_face_y, dprim_face_z,
                                             dim, nfp, neq, fp, gradPrim_x, gradPrim_y,
                                             gradPrim_z);
      Theseus::Kernels::el_gather_state(u_face_d, nfp, neq, fp, state1);
    
      const mfem::real_t ws = \
        Theseus::BC::ApplyViscousBoundaryCondition(dc, bc, state1, gradPrim_x, gradPrim_y,
                                                   gradPrim_z, nor_point, fluxN);
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

  template<typename PhysicsT>
  mfem::real_t NSOperator<PhysicsT>::MultCNS_Volume(const mfem::Vector &pu, const std::vector<mfem::Vector *> &p_grad_prim,
                                                    mfem::Vector &pdudt) const
  {
    Theseus::ScopedTimer timer("MultCNS_Volume");
    // Copy the device cache so that it is not member data
    auto dc = device_cache;
    const int dim = dc.dim;
    const int restr_size = operator_cache.restr_v->Height();

    mfem::Vector &vol_u(operator_cache.uVol);
    if(vol_u.Size() != restr_size){
      vol_u.SetSize(restr_size);
      vol_u.UseDevice();
    }

    mfem::Vector &dUe(operator_cache.rhsVol);
    if(dUe.Size() != restr_size){
      dUe.SetSize(restr_size);
      dUe.UseDevice();
    }

    std::vector<mfem::Vector> &vol_grad_prim(operator_cache.gradVol);
    if(vol_grad_prim.size() != dim){
      vol_grad_prim.resize(dim);
      for(int idim = 0;idim < dim;idim++){
        vol_grad_prim[idim].SetSize(restr_size);
        vol_grad_prim[idim].UseDevice();
      }
    }

    operator_cache.restr_v->Mult(pu, vol_u);
    for(int idim = 0;idim < dim;idim++){
      operator_cache.restr_v->Mult(*p_grad_prim[idim], vol_grad_prim[idim]);
    }

    // Zero the RHS array on-device
    {
      mfem::real_t *d = dUe.Write();
      mfem::forall(dUe.Size(), [=] MFEM_HOST_DEVICE (int i) { d[i] = mfem::real_t(0); });
    }

    // Set up the read-only pointers for restr inputs
    const mfem::real_t *Ue_d = vol_u.Read();
    const mfem::real_t *gradPrim_d[Theseus::MAXDIM] = {nullptr, nullptr, nullptr};
    for(int idim = 0;idim < dim;idim++){
      gradPrim_d[idim] = vol_grad_prim[idim].Read();
    }
  
    // Write-only for RHS
    mfem::real_t *dUe_d = dUe.Write();

    // Device cache parameters
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

      // Element-specific inputs and outputs
      const int eoff = e * estride;
      const mfem::real_t *u_el = Ue_d + eoff;
      mfem::real_t *du_el = dUe_d + eoff;

      mfem::real_t cs_el = \
        Theseus::DGSEMIntegrator::AssembleElementVolumeKernel(dc, u_el,
                                                              jac_el, metric_el, du_el);
#ifdef SUBCELL_FV_BLENDING
      mfem::real_t alpha_fv = alpha_d[e];
      if(alpha_fv > 1e-16){
        mfem::real_t alpha_dg = (1.0 - alpha_fv);
        mfem::real_t *du_fv = dUfv_d + eoff;
        const mfem::real_t *el_metric_xi = metric_xi_d + e * npe_metric_xi * dim;
        const mfem::real_t *el_metric_eta = (dim > 1 ? metric_eta_d + e * npe_metric_eta * dim :
                                             nullptr);
        const mfem::real_t *el_metric_zeta = (dim > 2 ? metric_zeta_d + e * npe_metric_zeta * dim :
                                              nullptr);
        const mfem::real_t cs_fv =                                              \
          Theseus::DGSEMIntegrator::ComputeFVFluxesKernel(dc, u_el, jac_el, el_metric_xi, el_metric_eta, el_metric_zeta, du_fv);
      
        for(int ipt = 0;ipt < estride;ipt++){
          du_el[ipt] = alpha_dg * du_el[ipt] + alpha_fv * du_fv[ipt];
        }
      
        cs_el = Kernels::rmax(cs_el, cs_fv);
      }
#endif
      ws_d[e] = cs_el;

      // Inviscid part is done: dUe currrently holds the inviscid RHS
      // Host code mixes inviscid and viscous assembly, we need separate.
      // Call the Viscous Assembly routine
      const mfem::real_t *grad_prim_el[Theseus::MAXDIM] = {nullptr, nullptr, nullptr};
      for(int idim = 0;idim < dim;idim++){
        grad_prim_el[idim] = gradPrim_d[idim] + eoff;
      }

      Theseus::DGSEMIntegrator::AssembleViscousElementVolumeKernel(dc, u_el, jac_el, metric_el,
                                                                   grad_prim_el[0], grad_prim_el[1],
                                                                   grad_prim_el[2], du_el);
    
    });
  
    // The rest is identical to Euler operator
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
  mfem::real_t NSOperator<PhysicsT>::MultCNS(const mfem::Vector &u, const std::vector<mfem::Vector *> &grad_prim,
                                             mfem::Vector &dudt) const
  {
    const int dim = operator_cache.dim;
    const mfem::Vector &pu = this->Prolongate(u);
    std::vector<mfem::Vector *> p_grad_(dim);
    if (this->P)
      {
        const int psize = this->P->Height();
        if(operator_cache.pGrad.size() != dim){
          operator_cache.pGrad.resize(dim);
          for(int idim = 0;idim < dim;idim++){
            operator_cache.pGrad[idim].SetSize(psize);
            operator_cache.pGrad[idim].UseDevice();
          }
        }
        for(int idim = 0;idim < dim;idim++){
          p_grad_[idim] = &(operator_cache.pGrad[idim]);
          this->P->Mult(*grad_prim[idim], *p_grad_[idim]);
        }
        if(operator_cache.pdudt.Size() != psize){
          operator_cache.pdudt.SetSize(psize);
          operator_cache.pdudt.UseDevice();
        }
      }
    const std::vector<mfem::Vector *> &pGradPrim = this->P ? p_grad_ : grad_prim;
    mfem::Vector &pdudt = this->P ? operator_cache.pdudt : dudt;
    pdudt = 0.0;
  
    mfem::real_t max_char_speed = MultCNS_Volume(pu, pGradPrim, pdudt);

    mfem::real_t max_char_speed_faces = MultCNS_InteriorFaces(pu, pGradPrim, pdudt);
    max_char_speed = std::max(max_char_speed, max_char_speed_faces);

    mfem::real_t max_char_speed_bnd = MultCNS_BoundaryFaces(pu, pGradPrim, pdudt);
    max_char_speed = std::max(max_char_speed, max_char_speed_bnd);
  
    if (this->Serial())
      {
        if (this->cP)
          {
            this->cP->MultTranspose(pdudt, dudt);
          }

      }
    else
      {
        this->P->MultTranspose(pdudt, dudt);
      }

    const int N = this->ess_tdof_list.Size();
    const auto idx = this->ess_tdof_list.Read();
    auto DU_RW = dudt.ReadWrite();
    mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) { DU_RW[idx[i]] = 0.0; });

    return max_char_speed;
  }

}
