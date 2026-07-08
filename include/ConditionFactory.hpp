// Copyright (c) 2025-2026 Board of Trustees of the University of Illinois
//
// This file is part of Theseus.
//
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <map>
#include <string>
#include <functional>
#include <iostream>

#include "mfem.hpp"

namespace Prandtl
{

  // Aliases for an initial condition function.
  using IC_Function0 = std::function<std::function<void(const mfem::Vector&, mfem::Vector&)>()>;
  using IC_Function1 = std::function<std::function<void(const mfem::Vector&, mfem::Vector&)>(mfem::real_t)>;
  using IC_Function2 = std::function<std::function<void(const mfem::Vector&, mfem::Vector&)>(mfem::real_t, mfem::real_t)>;
  using IC_Function3 = std::function<std::function<void(const mfem::Vector&,
                                                        mfem::Vector&)>(mfem::real_t, mfem::real_t, mfem::real_t)>;
  using IC_Function4 = std::function<std::function<void(const mfem::Vector&,
                                                        mfem::Vector&)>(mfem::real_t, mfem::real_t,
                                                                        mfem::real_t, mfem::real_t)>;
  using IC_Function5 = std::function<std::function<void(const mfem::Vector&,
                                                        mfem::Vector&)>(mfem::real_t, mfem::real_t, mfem::real_t,
                                                                        mfem::real_t, mfem::real_t)>;

  // Aliases for boundary condition function functions, vectors, and scalars.
  using BC_VectorFunction0 = std::function<std::function<void(const mfem::Vector&, mfem::Vector&)>()>;
  using BC_VectorFunction1 = std::function<std::function<void(const mfem::Vector&, mfem::Vector&)>(mfem::real_t)>;
  using BC_VectorFunction2 = std::function<std::function<void(const mfem::Vector&, mfem::Vector&)>(mfem::real_t, mfem::real_t)>;

  using BC_ScalarFunction0 = std::function<std::function<mfem::real_t(const mfem::Vector&)>()>;
  using BC_ScalarFunction1 = std::function<std::function<mfem::real_t(const mfem::Vector&)>(mfem::real_t)>;
  using BC_ScalarFunction2 = std::function<std::function<mfem::real_t(const mfem::Vector&)>(mfem::real_t, mfem::real_t)>;

  using BC_VectorTDFunction0 = std::function<std::function<void(const mfem::Vector&, mfem::real_t, mfem::Vector&)>()>;
  using BC_VectorTDFunction1 = std::function<std::function<void(const mfem::Vector&, mfem::real_t,
                                                                mfem::Vector&)>(mfem::real_t)>;
  using BC_VectorTDFunction2 = std::function<std::function<void(const mfem::Vector&, mfem::real_t,
                                                                mfem::Vector&)>(mfem::real_t, mfem::real_t)>;

  using BC_ScalarTDFunction0 = std::function<std::function<mfem::real_t(const mfem::Vector&, mfem::real_t)>()>;
  using BC_ScalarTDFunction1 = std::function<std::function<mfem::real_t(const mfem::Vector&, mfem::real_t)>(mfem::real_t)>;
  using BC_ScalarTDFunction2 = std::function<std::function<mfem::real_t(const mfem::Vector&,
                                                                        mfem::real_t)>(mfem::real_t, mfem::real_t)>;

  using BC_Scalar = mfem::real_t;
  using BC_Vector = mfem::Vector;


  class ConditionFactory
  {
  public:
    static ConditionFactory &Instance()
    {
      static ConditionFactory instance;
      return instance;
    }

    // Registration of initial condition function no parameters.
    void RegisterInitialCondition0(const std::string &key, IC_Function0 func)
    {
      initialConditions0_[key] = func;
    }

    // Registration of initial condition function with one parameter
    void RegisterInitialCondition1(const std::string &key, IC_Function1 func)
    {
      initialConditions1_[key] = func;
    }

    // Registration of initial condition function with two parameters
    void RegisterInitialCondition2(const std::string &key, IC_Function2 func)
    {
      initialConditions2_[key] = func;
    }

    // Registration of initial condition function with three parameters
    void RegisterInitialCondition3(const std::string &key, IC_Function3 func)
    {
      initialConditions3_[key] = func;
    }

    // Registration of initial condition function with four parameters
    void RegisterInitialCondition4(const std::string &key, IC_Function4 func)
    {
      initialConditions4_[key] = func;
    }

    // Registration of initial condition function with five parameters
    void RegisterInitialCondition5(const std::string &key, IC_Function5 func)
    {
      initialConditions5_[key] = func;
    }

    // Lookup initial condition function
    IC_Function0 GetInitialCondition0(const std::string &key)
    {
      return initialConditions0_.at(key);
    }

    // Lookup initial condition function with one parameter
    IC_Function1 GetInitialCondition1(const std::string &key)
    {
      return initialConditions1_.at(key);
    }

    // Lookup initial condition function with two parameters
    IC_Function2 GetInitialCondition2(const std::string &key)
    {
      return initialConditions2_.at(key);
    }

    // Lookup initial condition function with three parameters
    IC_Function3 GetInitialCondition3(const std::string &key)
    {
      return initialConditions3_.at(key);
    }

    // Lookup initial condition function with four parameters
    IC_Function4 GetInitialCondition4(const std::string &key)
    {
      return initialConditions4_.at(key);
    }

    // Lookup initial condition function with five parameters
    IC_Function5 GetInitialCondition5(const std::string &key)
    {
      return initialConditions5_.at(key);
    }

    // Registration of vector function boundary conditions with no parameters
    void RegisterVectorFunctionBoundaryCondition0(const std::string &key, BC_VectorFunction0 func)
    {
      boundaryConditionsVectorFunction0_[key] = func;
    }

    // Registration of scalar function boundary conditions with no parameters
    void RegisterScalarFunctionBoundaryCondition0(const std::string &key, BC_ScalarFunction0 func)
    {
      boundaryConditionsScalarFunction0_[key] = func;
    }

    // Registration of vector function boundary conditions with one parameter
    void RegisterVectorFunctionBoundaryCondition1(const std::string &key, BC_VectorFunction1 func)
    {
      boundaryConditionsVectorFunction1_[key] = func;
    }

    // Registration of scalar function boundary conditions with one parameter
    void RegisterScalarFunctionBoundaryCondition1(const std::string &key, BC_ScalarFunction1 func)
    {
      boundaryConditionsScalarFunction1_[key] = func;
    }

    // Registration of vector function boundary conditions with two parameters
    void RegisterVectorFunctionBoundaryCondition2(const std::string &key, BC_VectorFunction2 func)
    {
      boundaryConditionsVectorFunction2_[key] = func;
    }

    // Registration of scalar function boundary conditions with two parameters
    void RegisterScalarFunctionBoundaryCondition2(const std::string &key, BC_ScalarFunction2 func)
    {
      boundaryConditionsScalarFunction2_[key] = func;
    }

    // Registration of vector time-dependent function boundary conditions with no parameters
    void RegisterVectorTDFunctionBoundaryCondition0(const std::string &key, BC_VectorTDFunction0 func)
    {
      boundaryConditionsVectorTDFunction0_[key] = func;
    }

    // Registration of scalar time-dependent function boundary conditions with no parameters
    void RegisterScalarTDFunctionBoundaryCondition0(const std::string &key, BC_ScalarTDFunction0 func)
    {
      boundaryConditionsScalarTDFunction0_[key] = func;
    }

    // Registration of vector time-dependent function boundary conditions with one parameter
    void RegisterVectorTDFunctionBoundaryCondition1(const std::string &key, BC_VectorTDFunction1 func)
    {
      boundaryConditionsVectorTDFunction1_[key] = func;
    }

    // Registration of scalar time-dependent function boundary conditions with one parameter
    void RegisterScalarTDFunctionBoundaryCondition1(const std::string &key, BC_ScalarTDFunction1 func)
    {
      boundaryConditionsScalarTDFunction1_[key] = func;
    }
    
    // Registration of vector time-dependent function boundary conditions with two parameters
    void RegisterVectorTDFunctionBoundaryCondition2(const std::string &key, BC_VectorTDFunction2 func)
    {
      boundaryConditionsVectorTDFunction2_[key] = func;
    }

    // Registration of scalar time-dependent function boundary conditions with two parameters
    void RegisterScalarTDFunctionBoundaryCondition2(const std::string &key, BC_ScalarTDFunction2 func)
    {
      boundaryConditionsScalarTDFunction2_[key] = func;
    }

    // Registration of vector boundary conditions
    void RegisterVectorBoundaryCondition(const std::string &key, BC_Vector vector)
    {
      boundaryConditionsVector_[key] = vector;
    }

    // Registration of scalar boundary conditions
    void RegisterScalarBoundaryCondition(const std::string &key, BC_Scalar scalar)
    {
      boundaryConditionsScalar_[key] = scalar;
    }

    // Lookup vector functionboundary condition with no parameters
    BC_VectorFunction0 GetVectorFunctionBoundaryCondition0(const std::string &key)
    {
      return boundaryConditionsVectorFunction0_.at(key);
    }

    // Lookup scalar function boundary condition with no parameters
    BC_ScalarFunction0 GetScalarFunctionBoundaryCondition0(const std::string &key)
    {
      return boundaryConditionsScalarFunction0_.at(key);
    }

    // Lookup vector function boundary condition with one parameter
    BC_VectorFunction1 GetVectorFunctionBoundaryCondition1(const std::string &key)
    {
      return boundaryConditionsVectorFunction1_.at(key);
    }

    // Lookup scalar function boundary condition with one parameter
    BC_ScalarFunction1 GetScalarFunctionBoundaryCondition1(const std::string &key)
    {
      return boundaryConditionsScalarFunction1_.at(key);
    }

    // Lookup vector function boundary condition with two parameters
    BC_VectorFunction2 GetVectorFunctionBoundaryCondition2(const std::string &key)
    {
      return boundaryConditionsVectorFunction2_.at(key);
    }

    // Lookup scalar function boundary condition with two parameters
    BC_ScalarFunction2 GetScalarFunctionBoundaryCondition2(const std::string &key)
    {
      return boundaryConditionsScalarFunction2_.at(key);
    }

    // Lookup vector time-dependent function boundary condition with no parameters
    BC_VectorTDFunction0 GetVectorTDFunctionBoundaryCondition0(const std::string &key)
    {
      return boundaryConditionsVectorTDFunction0_.at(key);
    }

    // Lookup scalar time-dependent function boundary condition with no parameters
    BC_ScalarTDFunction0 GetScalarTDFunctionBoundaryCondition0(const std::string &key)
    {
      return boundaryConditionsScalarTDFunction0_.at(key);
    }

    // Lookup vector time-dependent function boundary condition with one parameter
    BC_VectorTDFunction1 GetVectorTDFunctionBoundaryCondition1(const std::string &key)
    {
      return boundaryConditionsVectorTDFunction1_.at(key);
    }

    // Lookup scalar time-dependent function boundary condition with one parameter
    BC_ScalarTDFunction1 GetScalarTDFunctionBoundaryCondition1(const std::string &key)
    {
      return boundaryConditionsScalarTDFunction1_.at(key);
    }

    // Lookup vector time-dependent function boundary condition with two parameters
    BC_VectorTDFunction2 GetVectorTDFunctionBoundaryCondition2(const std::string &key)
    {
      return boundaryConditionsVectorTDFunction2_.at(key);
    }

    // Lookup scalar time-dependent function boundary condition with two parameters
    BC_ScalarTDFunction2 GetScalarTDFunctionBoundaryCondition2(const std::string &key)
    {
      return boundaryConditionsScalarTDFunction2_.at(key);
    }

    // Lookup vector boundary condition
    BC_Vector GetVectorBoundaryCondition(const std::string &key)
    {
      return boundaryConditionsVector_.at(key);
    }

    // Lookup scalar boundary condition
    BC_Scalar GetScalarBoundaryCondition(const std::string &key)
    {
      return boundaryConditionsScalar_.at(key);
    }


  private:
    ConditionFactory() {}

    std::map<std::string, IC_Function0> initialConditions0_;
    std::map<std::string, IC_Function1> initialConditions1_;
    std::map<std::string, IC_Function2> initialConditions2_;
    std::map<std::string, IC_Function3> initialConditions3_;
    std::map<std::string, IC_Function4> initialConditions4_;
    std::map<std::string, IC_Function5> initialConditions5_;

    std::map<std::string, BC_VectorFunction0> boundaryConditionsVectorFunction0_;
    std::map<std::string, BC_VectorFunction1> boundaryConditionsVectorFunction1_;
    std::map<std::string, BC_VectorFunction2> boundaryConditionsVectorFunction2_;

    std::map<std::string, BC_ScalarFunction0> boundaryConditionsScalarFunction0_;
    std::map<std::string, BC_ScalarFunction1> boundaryConditionsScalarFunction1_;
    std::map<std::string, BC_ScalarFunction2> boundaryConditionsScalarFunction2_;

    std::map<std::string, BC_VectorTDFunction0> boundaryConditionsVectorTDFunction0_;
    std::map<std::string, BC_VectorTDFunction1> boundaryConditionsVectorTDFunction1_;
    std::map<std::string, BC_VectorTDFunction2> boundaryConditionsVectorTDFunction2_;

    std::map<std::string, BC_ScalarTDFunction0> boundaryConditionsScalarTDFunction0_;
    std::map<std::string, BC_ScalarTDFunction1> boundaryConditionsScalarTDFunction1_;
    std::map<std::string, BC_ScalarTDFunction2> boundaryConditionsScalarTDFunction2_;

    std::map<std::string, BC_Vector> boundaryConditionsVector_;
    
    std::map<std::string, BC_Scalar> boundaryConditionsScalar_;
  
  };

} // namespace Theseus
