//                                PUMI Example
//
// Sample run:   ./read_pumi_mesh -p ../pumi-meshes/cube/cube.dmg  -m ../pumi-meshes/cube/pumi11/cube.smb
//
// Description:  The purpose of this example is to read a PUMI mesh and  model
//               and create an MFEM mesh object.

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
  int order = 1;
  bool visualization = 1;
  double alpha  = 0.25;
  double beta = 2.0;
  int n_target = 2;

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
  apf::Field* residualErrorField = pumi_mesh->getField(2);// 3 for dd antenna

  // 2. Compute Sizefield for mesh adaptation
  apf::Field* sizefield = em::getTargetEMSizeField(
      electric_field, residualErrorField, n_target, alpha, beta);
  cout << "Size field computed" << endl;

  // 3. Write Fields to PUMI
  apf::Field* residualErrorNodalField = 0;
  residualErrorNodalField = apf::createField(
      pumi_mesh, "residual_error_nodal_field", apf::SCALAR, apf::getLagrange(1));

  // Convert Lagrange Constant Fields to Nodal Fields
  apf::MeshEntity* ent;
  apf::MeshIterator* itr;
  itr = pumi_mesh->begin(0);
  while ((ent = pumi_mesh->iterate(itr)))
  {
    apf::Adjacent elements;
    pumi_mesh->getAdjacent(ent, 3, elements);
    int ne = (int) elements.getSize();

    // error nodal residual fields
    double residual_errors[ne];
    for (std::size_t i=0; i < ne; ++i) {
      residual_errors[i] = apf::getScalar(residualErrorField, elements[i], 0);
    }
    double residual_average = 0.0;
    for(int i = 0; i < ne; i++){
      residual_average += residual_errors[i];
    }
    residual_average = residual_average/ne;
    apf::setScalar(residualErrorNodalField, ent, 0, residual_average);
  }

  apf::writeVtkFiles("icrf_antenna_fields_pumi", pumi_mesh);
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
  pumi_mesh->writeNative("icrf_antenna_adapted.smb");
  apf::writeVtkFiles("icrf_antenna_adapted", pumi_mesh);
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


