//
// Sample run:   ./testNedelecMFEMPUMI -p ../pumi-meshes/cube/cube.dmg  -m ../pumi-meshes/cube/pumi11/cube.smb -o order
//
// Description:  The purpose of this example is to read a PUMI mesh and  model
//               and create an MFEM mesh object. Then, define variable order
//               Nedelec Fields on it. Check if fields on MFEM and PUMI match
//               when evaluated at a certain xi coordinate of an element.

#include "mfem.hpp"
#include <fstream>
#include <iostream>

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
#include "apf.h"
#include "apfField.h"
#include "apfShape.h"
#include "maTables.h"

using namespace std;
using namespace mfem;

// user defined functions
void E_exact(const Vector &, Vector &);
void A_const(const Vector &, Vector &);
void A_linear(const Vector &x, Vector &f);
void A_quadratic(const Vector &x, Vector &f);
void A_cubic(const Vector &x, Vector &f);

int dim;

int main(int argc, char *argv[])
{
  // initilize mpi
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
  int order = 2;
  bool visualization = 1;

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

  args.Parse();
  if (!args.Good())
  {
     args.PrintUsage(cout);
     return 1;
  }
  args.PrintOptions(cout);

  // 1. Read the SCOREC Mesh
  PCU_Comm_Init();
#ifdef MFEM_USE_SIMMETRIX
  //SimUtil_start();
  Sim_readLicenseFile(0);
  gmi_sim_start();
  gmi_register_sim();
#endif
  gmi_register_mesh();

  apf::Mesh2* pumi_mesh;
  pumi_mesh = apf::loadMdsMesh(model_file, mesh_file);


  // If it is higher order change shape
  if (order > 1){
      crv::BezierCurver bc(pumi_mesh, order, 2);
      bc.run();
  }
  pumi_mesh->verify();

  // 2. Create Nedelec Field on PUMI mesh.
  apf::Field* nedelecField = apf::createField(
              pumi_mesh, "nedelec_field", apf::SCALAR, apf::getNedelec(order));

  // 3. Create the MFEM mesh object from the PUMI mesh. We can handle triangular
  //    and tetrahedral meshes. Other inputs are the same as MFEM default
  //    constructor.
  ParPumiMesh* pmesh = new ParPumiMesh(MPI_COMM_WORLD, pumi_mesh);
  pmesh->ReorientTetMesh(); // *** for ND spaces
  dim = pmesh->Dimension();
  int sdim = pmesh->SpaceDimension();

  // 4. Define a finite element space on the mesh. Here we use the Nedelec
  //    finite elements of the specified order.
  FiniteElementCollection *fec = new ND_FECollection(order, dim);
  ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);

  // 5. Define a grid function 'gf' on Nedelec space.
  // Project the user function on the grid function.
  ParGridFunction gf(fespace);
  //VectorFunctionCoefficient E(sdim, E_exact);
  VectorFunctionCoefficient E(sdim, A_cubic);
  gf.ProjectCoefficient(E);

  // 6. Evaluate vector field using MFEM vector shape functions.
  // Note: we take care of effects of different orientations on
  // xi coordinates here.
  int elemNo = 0;
  apf::MeshEntity* ent;
  apf::MeshIterator* it;
  it = pumi_mesh->begin(dim);
  while ( ent = pumi_mesh->iterate(it) ) {

    // set xi coordinate
    apf::Vector3 xi = apf::Vector3(1./6., 1./4., 1./5.);
    apf::MeshElement* me = apf::createMeshElement(pumi_mesh, ent);
    apf::Vector3 global;
    apf::mapLocalToGlobal(me, xi, global);
    Vector tr (dim);
    tr[0] = global[0]; tr[1] = global[1]; tr[2] = global[2];
    ElementTransformation* eltr = pmesh->GetElementTransformation(elemNo);
    IntegrationPoint ip;
    eltr->TransformBack(tr, ip);

    // evaluate vector field at xi coordinate
    Vector mfem_field_vector;
    gf.GetVectorValue(elemNo, ip, mfem_field_vector);

    // evaluate exact field at physical coordinate
    eltr->Transform(ip, tr);
    Vector exact_field (dim);
    A_cubic(tr, exact_field);

    cout << "at tet " << elemNo << endl;
    cout << "MFEM interpolated field ";
    mfem_field_vector.Print(cout, mfem_field_vector.Size());
    cout << "MFEM        exact field ";
    exact_field.Print(cout, exact_field.Size());
    cout << "=======================================";
    cout << endl;

    apf::destroyMeshElement(me);

    elemNo++;
  }


  // 7. Project MFEM Field onto PUMI Field and store dofs
  //NedelecFieldMFEMtoPUMI(pmesh, pumi_mesh, &gf, nedelecField);
  pmesh->NedelecFieldMFEMtoPUMI(pumi_mesh, &gf, nedelecField);


  // 8. Evaluate vector field using PUMI vector shape functions
  //    Verify that interpolated PUMI solution field agrees with
  //    the interpolated MFEM solution field.
  apf::Vector3 xi = apf::Vector3(1./6., 1./4., 1./5.);
  int i = 0;
  it = pumi_mesh->begin(dim);
  while ( ent = pumi_mesh->iterate(it) ) {

    apf::MeshElement* me = apf::createMeshElement(pumi_mesh, ent); // map xi to global
    apf::Element* el = apf::createElement(nedelecField, me);
    apf::Vector3 pumi_field_vector;
    apf::getVector(el, xi, pumi_field_vector);

    cout << "at tet " << i << endl;
    std::cout << " PUMI interpolated field "<< pumi_field_vector << std::endl;
    cout << "=======================================";
    cout << endl;

    apf::destroyElement(el);
    apf::destroyMeshElement(me);

    i++;
  }
  cout << endl;


  delete pmesh;

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

void E_exact(const Vector &x, Vector &E)
{
   if (dim == 3)
   {
      E(0) = sin(M_PI * x(1));
      E(1) = sin(M_PI * x(2));
      E(2) = sin(M_PI * x(0));
   }
   else
   {
      E(0) = sin(M_PI * x(1));
      E(1) = sin(M_PI * x(0));
      if (x.Size() == 3) { E(2) = 0.0; }
   }
}

void A_const(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = 10.;
      f(1) = 20.;
      f(2) = 30.;
   }
   else
   {}
}

void A_linear(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = x[2];
      f(1) = x[1];
      f(2) = x[0];
   }
   else
   {}
}

void A_quadratic(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = x[2]*x[2];
      f(1) = x[1]*x[1];
      f(2) = x[0]*x[0];
   }
   else
   {}
}

void A_cubic(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = x[2]*x[2]*x[2];
      f(1) = x[1]*x[1]*x[1];
      f(2) = x[0]*x[0]*x[0];
   }
   else
   {}
}
