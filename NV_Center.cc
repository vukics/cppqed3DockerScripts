// Copyright András Vukics 2006–2023. Distributed under the Boost Software License, Version 1.0. (See accompanying file LICENSE.txt)

#include "Spin.h"
#include "Qbit.h"

#include "QuantumJumpMonteCarlo.h"
#include "Schrödinger.h"

using std::numbers::pi;
using namespace quantumtrajectory; using namespace qbit::multidiagonal;
using parameters::_;

using quantumoperator::multidiagonal::identity;

int main(int argc, char* argv[])
{
  auto op{optionParser()};

  trajectory::Pars<schrödinger::Pars> pt(op);
  // trajectory::Pars<qjmc::Pars<pcg64>> pt(op);

  double ge, gn, delta, Bz, Axx, Ayy, Azz, Azx, Bx0, omega, phi;

  add(op,"NV_Center",
      _("delta","zero field splitting (splits s_z = 0 and s_z = +1, -1)", 2*pi * 2880, delta),
      _("ge","electron gyromagnetic ratio", 2*pi * 28020, ge),
      _("gn","nuclear gyromagnetic ratio", 2*pi * 10.71, gn),
      _("Bz","static magnetic field", 0.0403, Bz),
      _("Axx", "hyperfine interaction tensor xx component", 2*pi * 0.170905, Axx),
      _("Ayy", "hyperfine interaction tensor yy component", 2*pi * 0.170905, Ayy),
      _("Azz", "hyperfine interaction tensor zz component", 2*pi * 0.213154, Azz),
      _("Azx", "hyperfine interaction tensor zx component",  2*pi * (-0.003), Azx),
      _("Bx0", "drive magnetic field amplitude", 0.1, Bx0),
      _("omega", "drive magnetic field frequency", 2*pi * 8, omega),
      _("phi", "drive magnetic field phase", 0., phi));

  parse(op,argc, argv);

  BinarySystem bs {
    QuantumSystemDynamics{2, SystemFrequencyStore{},Liouvillian<1>{},
                          (ge*Bz*sz() + delta*( sz() | sz() ) ) / 1.i , // freeElectronSpinHamiltonian
                          exact_propagator_ns::noOp, expectation_values_ns::noOp },
    QuantumSystemDynamics{3, SystemFrequencyStore{},Liouvillian<1>{},
                          gn*Bz*spin::sz(2) / 1.i , // freeNuclearSpinHamiltonian
                          exact_propagator_ns::noOp,expectation_values_ns::noOp},
    SystemFrequencyStore{{"omega",omega,1}}, // omega is assumed to be the largest frequency
    Liouvillian<2>{},
    makeHamiltonianCollection<2>(
      ( Axx*sx()*spin::sx(2) + Ayy*sy()*spin::sy(2) + Azz*sz()*spin::sz(2) + Azx*(sz()*spin::sx(2) + sx()*spin::sz(2)) ) / 1.i , // interactionHamiltonian
      [&, h=( ge*sx()*identity(3)+gn*identity(2)*spin::sx(2) ) / 1.i] (double t, StateVectorConstView<2> psi, StateVectorView<2> dpsidt)
      {
        double factor = Bx0*sin(omega*t + phi);
        (factor*h)(0.,psi,dpsidt);
      } // driveHamiltonian
    ),
    exact_propagator_ns::noOp,
    [] (LDO<StateVector,2> psi) {
      // std::cerr<<json(psi)<<std::endl;
      return hana::make_tuple(psi(0,0),psi(0,1),psi(0,2),psi(1,0),psi(1,1),psi(1,2));
    } /*expectation_values_ns::noOp*/};

  quantumdata::StateVector<2> psi{{2,3}}; // psi(0,0)=1;// psi(1)=1;

  run(
    schrödinger::make<cppqedutils::ODE_EngineBoost>(std::move(bs),std::move(psi),pt),
    pt,trajectory::observerNoOp);

}
