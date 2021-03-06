//                                PUMI Example
//
// Sample run:   ./read_pumi_mesh -p ../pumi-meshes/cube/cube.dmg  -m ../pumi-meshes/cube/pumi11/cube.smb
//
// Description:  The purpose of this example is to read a PUMI mesh and  model
//               and create an MFEM mesh object.

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

   //Read the SCOREC Mesh
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


   //If it is higher order change shape
   if (order > 1){
       crv::BezierCurver bc(pumi_mesh, order, 2);
       bc.run();
   }
   pumi_mesh->verify();

   // 2. Create the MFEM mesh object from the PUMI mesh. We can handle triangular
   //    and tetrahedral meshes. Other inputs are the same as MFEM default
   //    constructor.
   Mesh *mesh = new PumiMesh(pumi_mesh, 1, 1);

   delete mesh;

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
