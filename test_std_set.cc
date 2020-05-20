//                                PUMI Example
//
// Sample run:   ./test_std_set -p ../pumi-meshes/cube/cube.dmg  -m ../pumi-meshes/cube/pumi11/cube.smb
//
// Description:  The purpose of this example is to test if the mesh entities
//               inserted into the std::set are still at the same index when
//               they are retrieved. They are not because the std::set is
//               implemented using bindary trees and the values are not inserted
//               into it sequentially as they are in std::vector.
//               std::vector is included in the test file for comparison.

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

typedef std::set<apf::MeshEntity*> EntitySet;
typedef std::vector<apf::MeshEntity*> EntityVector;

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
  int dim = mesh->Dimension();


  // 3. Insert entities in EntitySet
  apf::MeshEntity* ent;
  apf::MeshIterator* it;
  it = pumi_mesh->begin(dim);
  int nf;
  EntitySet facesSet;
  EntityVector facesVector;
  while ( ent = pumi_mesh->iterate(it) ) {
    cout << "At tet 0" << endl;

    apf::Downward f;
    nf = pumi_mesh->getDownward(ent, 2, f);
    for (int i = 0; i < nf; i++) {
      cout << "Face " << i << endl;
      facesSet.insert(f[i]);
      facesVector.push_back(f[i]);

      apf::Downward v;
      int nv = pumi_mesh->getDownward(f[i], 0, v);

      apf::Vector3 p;
      for (int j = 0; j < nv; j++) {
        pumi_mesh->getPoint(v[j], 0, p);
        cout << p << endl;
      }
      cout << endl;
    }
    break;
  }
  pumi_mesh->end(it);


  // 4. Retrieve entities from EntitySet
  cout << endl << "Retrieving Entities from EntitySet" << endl;
  for (int i = 0; i < nf; i++) {
    EntitySet::iterator it = std::next(facesSet.begin(), i);
    apf::MeshEntity* face = *it;

    apf::Downward v;
    int nv = pumi_mesh->getDownward(face, 0, v);

    apf::Vector3 p;
    for (int j = 0; j < nv; j++) {
      pumi_mesh->getPoint(v[j], 0, p);
      cout << p << endl;
    }
    cout << endl;
  }

  // 4. Retrieve entities from EntityVector
  cout << endl << "Retrieving Entities from EntityVector" << endl;
  for (int i = 0; i < nf; i++) {

    apf::Downward v;
    int nv = pumi_mesh->getDownward(facesVector[i], 0, v);

    apf::Vector3 p;
    for (int j = 0; j < nv; j++) {
      pumi_mesh->getPoint(v[j], 0, p);
      cout << p << endl;
    }
    cout << endl;
  }


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
