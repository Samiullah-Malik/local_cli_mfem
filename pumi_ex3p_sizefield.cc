//                                PUMI Example
//
// Sample run:
// /pumi_ex3p_sizefield -m fichera_error_pumi.smb -p /users/maliks2/Meshes/MaxwellMeshes/Fichera/25k/fichera_25k_nat.x_t -o 1 -a 0.0001 -b 20.0 -n 1 > pumi
//
// Description:  The purpose of this example is to compute size field on a PUMI
//               mesh using the per-element error field on it.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include <lionPrint.h>
#ifdef MFEM_USE_SIMMETRIX
#include <SimUtil.h>
#include <gmi_sim.h>
#endif
#include <apfMDS.h>
#include <gmi_null.h>
#include <PCU.h>
#include <apfConvert.h>
#include <gmi_mesh.h>
#include <crv.h>
#include <em.h>

using namespace std;
using namespace mfem;

void pumiUserFunction(const apf::Vector3& x, apf::Vector3& f);
void E_exact(const apf::Vector3 &x, apf::Vector3& E);
double computeElementExactError(apf::Mesh* mesh, apf::MeshEntity* e,
  apf::Field* f);

double freq = 1.0, kappa;
int dim;
int order = 1;
bool visualization = 1;
double alpha  = 0.25;
double beta = 2.0;
int n_target = 2;

int main(int argc, char *argv[])
{
   //initilize mpi
   int num_proc, myId;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
   MPI_Comm_rank(MPI_COMM_WORLD, &myId);

  // 1. Parse command-line options.
  const char *mesh_file = "../data/pumi/serial/sphere.smb";
#ifdef MFEM_USE_SIMMETRIX
  const char *model_file = "../data/pumi/geom/sphere.x_t";
#else
  const char *model_file = "../data/pumi/geom/sphere.dmg";
#endif

  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&model_file, "-p", "--parasolid",
                 "Parasolid model to use.");
  args.AddOption(&alpha , "-a", "--alpha",
                 "alpha");
  args.AddOption(&beta , "-b", "--beta",
                 "beta");
  args.AddOption(&n_target , "-n", "--n_target",
                 "n_target");

  args.Parse();
  if (!args.Good())
  {
     args.PrintUsage(cout);
     return 1;
  }
  args.PrintOptions(cout);
  kappa = freq * M_PI;

  //Read the SCOREC Mesh
  PCU_Comm_Init();
  lion_set_verbosity(1);
#ifdef MFEM_USE_SIMMETRIX
  //SimUtil_start();
  Sim_readLicenseFile(0);
  gmi_sim_start();
  gmi_register_sim();
#endif
  gmi_register_mesh();

  apf::Mesh2* pumi_mesh;
  pumi_mesh = apf::loadMdsMesh(model_file, mesh_file);
  dim = pumi_mesh->getDimension();

  //If it is higher order change shape
  if (order > 1){
      crv::BezierCurver bc(pumi_mesh, order, 2);
      bc.run();
  }
  pumi_mesh->verify();

  cout << "Initial Fields " << pumi_mesh->countFields() << endl;
  int i = 0;
  while(i < pumi_mesh->countFields()) {
    apf::Field* f = pumi_mesh->getField(i++);
    cout << apf::getName(f) << endl;
  }
  apf::Field* electric_field = pumi_mesh->getField(0);
  apf::Field* residualErrorField = pumi_mesh->getField(1);

  // 2. Compute Sizefield for mesh adaptation
  apf::Field* sizefield = em::getTargetEMSizeField(
      electric_field, residualErrorField, n_target, alpha, beta);
  cout << "Size field computed" << endl;

  // 3. Write Fields to PUMI
  apf::Field* E_exact_Field = 0;
  apf::Field* E_exact_nodal_Field = 0;
  apf::Field* E_fem_Field = 0;
  apf::Field* E_fem_nodal_Field = 0;

  apf::Field* exactErrorField = 0;
  apf::Field* exactErrorNodalField = 0;
  apf::Field* residualErrorNodalField = 0;

  E_exact_Field = apf::createIPField(
      pumi_mesh, "E_exact_field", apf::VECTOR, 1);
  E_exact_nodal_Field = apf::createField(
      pumi_mesh, "E_exact_nodal_field", apf::VECTOR, apf::getLagrange(1));
  E_fem_Field = apf::createIPField(
      pumi_mesh, "E_fem_field", apf::VECTOR, 1);
  E_fem_nodal_Field = apf::createField(
      pumi_mesh, "E_fem_nodal_field", apf::VECTOR, apf::getLagrange(1));
  exactErrorField = apf::createIPField(
      pumi_mesh, "exact_error_field", apf::SCALAR, 1);
  exactErrorNodalField = apf::createField(
      pumi_mesh, "exact_error_nodal_field", apf::SCALAR, apf::getLagrange(1));
  residualErrorNodalField = apf::createField(
      pumi_mesh, "residual_error_nodal_field", apf::SCALAR, apf::getLagrange(1));


  // Write electric fields (fem and exact)
  apf::MeshEntity* ent;
  apf::MeshIterator* itr = pumi_mesh->begin(3);
  while ((ent = pumi_mesh->iterate(itr)))
  {
    apf::Vector3 c = apf::Vector3(1./3.,1./3.,1./3.);

    apf::MeshElement* me = apf::createMeshElement(pumi_mesh, ent);
    apf::Element* el = apf::createElement(electric_field, me);

    // fem field (Vector)
    apf::Vector3 vvalue;
    apf::getVector(el, c, vvalue);
    apf::setVector(E_fem_Field, ent, 0, vvalue);

    // exact field (Vector)
    apf::Vector3 cg;
    apf::mapLocalToGlobal(me, c, cg);
    E_exact(cg, vvalue);
    apf::setVector(E_exact_Field, ent, 0, vvalue);

    apf::destroyElement(el);
    apf::destroyMeshElement(me);
  }
  pumi_mesh->end(itr);

  // Write exact error field
  itr = pumi_mesh->begin(3);
  while ((ent = pumi_mesh->iterate(itr)))
  {
    double exact_element_error = computeElementExactError(
        pumi_mesh, ent, electric_field);
    apf::setScalar(exactErrorField, ent, 0, exact_element_error);
  }
  pumi_mesh->end(itr);

  // Write Nodal Fields
  itr = pumi_mesh->begin(0);
  while ((ent = pumi_mesh->iterate(itr)))
  {
    apf::Adjacent elements;
    pumi_mesh->getAdjacent(ent, 3, elements);
    int ne = (int) elements.getSize();

    // nodal error fields (exact, residual)
    double exact_errors[ne];
    double residual_errors[ne];
    for (std::size_t i=0; i < ne; ++i) {
      exact_errors[i] = apf::getScalar(exactErrorField, elements[i], 0);
      residual_errors[i] = apf::getScalar(residualErrorField, elements[i], 0);
    }
    double exact_average = 0.0;
    double residual_average = 0.0;
    for(int i = 0; i < ne; i++){
      exact_average += exact_errors[i];
      residual_average += residual_errors[i];
    }
    exact_average = exact_average/ne;
    residual_average = residual_average/ne;

    apf::setScalar(exactErrorNodalField, ent, 0, exact_average);
    apf::setScalar(residualErrorNodalField, ent, 0, residual_average);

    // projected nodal electric fields (exact, FEM)
    apf::Vector3 exact_field_avg, exact_field;
    exact_field_avg.zero();
    for (std::size_t i=0; i < ne; ++i){
      apf::getVector(E_exact_Field, elements[i], 0, exact_field);
      exact_field_avg += exact_field;
    }
    exact_field_avg = exact_field_avg * (1./ne);
    apf::setVector(E_exact_nodal_Field, ent, 0, exact_field_avg);

    apf::Vector3 fem_field_avg, fem_field;
    fem_field_avg.zero();
    for (std::size_t i=0; i < ne; ++i){
      apf::getVector(E_fem_Field, elements[i], 0, fem_field);
      fem_field_avg += fem_field;
    }
    fem_field_avg = fem_field_avg * (1./ne);
    apf::setVector(E_fem_nodal_Field, ent, 0, fem_field_avg);
  }
  pumi_mesh->end(itr);

  apf::writeVtkFiles("fichera_fields_pumi", pumi_mesh);
  cout << "Fields Written" << endl;

  // 4. Perform Mesh Adapt
  int index = 0;
  while (pumi_mesh->countFields() > 1)
  {
    apf::Field* f = pumi_mesh->getField(index);
    if (f == sizefield) {
       index++;
       continue;
    }
    pumi_mesh->removeField(f);
    apf::destroyField(f);
  }
  pumi_mesh->verify();

  ma::Input* erinput = ma::configure(pumi_mesh, sizefield);
  erinput->shouldFixShape = true;
  erinput->maximumIterations = 10;
  ma::adapt(erinput);
  pumi_mesh->writeNative("fichera_adapted.smb");
  apf::writeVtkFiles("fichera_adapted", pumi_mesh);
  cout << "Finished Mesh Adapt " << endl;

  pumi_mesh->destroyNative();
  apf::destroyMesh(pumi_mesh);
  PCU_Comm_Free();

#ifdef MFEM_USE_SIMMETRIX
  gmi_sim_stop();
  Sim_unregisterAllKeys();
  //SimUtil_stop();
#endif

  MPI_Finalize();
  return 0;
}

// f_exact (forcing function)
void pumiUserFunction(const apf::Vector3& x, apf::Vector3& f)
{
  if (dim == 3)
  {
      f[0] = (1. + kappa * kappa) * sin(kappa * x[1]);
      f[1] = (1. + kappa * kappa) * sin(kappa * x[2]);
      f[2] = (1. + kappa * kappa) * sin(kappa * x[0]);
  }
  else
  {
     f[0] = (1. + kappa * kappa) * sin(kappa * x[1]);
     f[1] = (1. + kappa * kappa) * sin(kappa * x[0]);
     f[2] = 0.0;
  }
}
void E_exact(const apf::Vector3 &x, apf::Vector3& E)
{
   if (dim == 3)
   {
      E[0] = sin(kappa * x[1]);
      E[1] = sin(kappa * x[2]);
      E[2] = sin(kappa * x[0]);
   }
   else
   {
      E[0] = sin(kappa * x[1]);
      E[1] = sin(kappa * x[0]);
      E[2] = 0.0;
   }
}

double computeElementExactError(apf::Mesh* mesh, apf::MeshEntity* e,
  apf::Field* f)
{
  double error = 0.0;

  apf::FieldShape* fs = f->getShape();
  int type = mesh->getType(e);
  PCU_ALWAYS_ASSERT(type == apf::Mesh::TET);
  int nd = apf::countElementNodes(fs, type);
  int dim = apf::getDimension(mesh, e);
  double w;

  apf::MeshElement* me = apf::createMeshElement(mesh, e);
  apf::Element* el = apf::createElement(f, me);
  int order = 2*fs->getOrder() + 1;
  int np = apf::countIntPoints(me, order);

  apf::Vector3 femsol, exsol;

  apf::Vector3 p;
  for (int i = 0; i < np; i++) {
    apf::getIntPoint(me, order, i, p);
    double weight = apf::getIntWeight(me, order, i);
    apf::Matrix3x3 J;
    apf::getJacobian(me, p, J);
    double jdet = apf::getJacobianDeterminant(J, dim);
    w = weight * jdet;

    apf::getVector(el, p, femsol);

    apf::Vector3 global;
    apf::mapLocalToGlobal(me, p, global);
    E_exact(global, exsol);
    apf::Vector3 diff = exsol - femsol;

    error += w * (diff * diff);
  }
  if (error < 0.0)
    error = -error;

  return sqrt(error);
}
