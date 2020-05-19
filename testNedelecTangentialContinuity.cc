//                                PUMI Example
//
// Sample run: ./testNedelecTangentialContinuity -p ../pumi-meshes/cube/cube.dmg  -m ../pumi-meshes/cube/pumi11/cube.smb/
//
// Description:  The purpose of this example is to test the tangential continuity
//               of Nedelec vector shape functions across an inter-element
//               boundary.
//
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
   int dim = mesh->Dimension();

  // 3. Define ND Space on MFEM mesh
  FiniteElementCollection *fec = new ND_FECollection(order, dim);
  FiniteElementSpace *fespace = new FiniteElementSpace(mesh, fec);

  // 4. Test Tangential Continuity of Nedelec Vector Shape Functions
  for (int i = 0; i < mesh->GetNFaces(); i++) {
    FaceElementTransformations* trans = mesh->GetFaceElementTransformations(i);

    //Pick a Face containing two upward adjacent tet elements
    if ((trans->Elem1No != -1) && (trans->Elem2No != -1)) {

      // elem1no
      Geometry::Type geo_type = mesh->GetFaceGeometryType(trans->Face->ElementNo);
      const IntegrationPoint *center = &Geometries.GetCenter(geo_type); // center of face
      trans->Face->SetIntPoint(center);

      Vector normal(3); normal = 0.0;
      CalcOrtho(trans->Face->Jacobian(), normal); // compute normal
      normal /= normal.Norml2();
      IntegrationPoint eip1;
      trans->Loc1.Transform(*center, eip1);
      trans->Elem1->SetIntPoint(&eip1);

      const FiniteElement* fe1 = fespace->GetFE(trans->Elem1No);
      DenseMatrix shape_mat(fe1->GetDof(), fe1->GetDim()); // dof x dim
      fe1->CalcVShape(*trans->Elem1, shape_mat);

      // negating shape functions of entities with negative orientations
      Array<int> eldofs; fespace->GetElementDofs(trans->Elem1No, eldofs);
      for(int d = 0; d < eldofs.Size(); d++) {
        if(eldofs[d] < 0) {
          for(int j = 0; j < dim; j++)
            {shape_mat(d,j) = -1*shape_mat(d,j);}
        }
      }

      // compute tangential components of element shape functions
      // evaluated at center of the face
      Vector shapes_mult_normal (fe1->GetDof());
      shape_mat.Mult(normal, shapes_mult_normal);
      DenseMatrix shapes_normal(fe1->GetDof(), fe1->GetDim());
      for (int i = 0; i < fe1->GetDof(); i++) {
        Vector shape_nc (normal);
        shape_nc *= shapes_mult_normal(i);
        shapes_normal.SetRow(i, shape_nc);
      }

      DenseMatrix shapes_tangent(fe1->GetDof(), fe1->GetDim());
      shapes_tangent += shape_mat;
      shapes_tangent -= shapes_normal;


      cout <<"Tangential Components of Element 1 Shape Functions" << endl;
      shapes_tangent.Print(cout, dim);
      cout << "===================" << endl;

      // elem2no
      IntegrationPoint eip2;
      trans->Loc2.Transform(*center, eip2);
      trans->Elem2->SetIntPoint(&eip2);

      normal.Neg();

      const FiniteElement* fe2 = fespace->GetFE(trans->Elem2No);
      fe2->CalcVShape(*trans->Elem2, shape_mat);

      // negating shape functions of entities with negative orientations
      fespace->GetElementDofs(trans->Elem2No, eldofs);
      for(int d = 0; d < eldofs.Size(); d++) {
        if(eldofs[d] < 0) {
          for(int j = 0; j < dim; j++)
            {shape_mat(d,j) = -1*shape_mat(d,j);}
        }
      }

      // compute tangential components of element shape functions
      // evaluated at center of the face
      shapes_mult_normal = 0.0;
      shape_mat.Mult(normal, shapes_mult_normal);
      shapes_normal = 0.0;
      for (int i = 0; i < fe2->GetDof(); i++) {
        Vector shape_nc (normal);
        shape_nc *= shapes_mult_normal(i);
        shapes_normal.SetRow(i, shape_nc);
      }

      shapes_tangent = 0.0;
      shapes_tangent += shape_mat;
      shapes_tangent -= shapes_normal;

      cout <<"Tangential Components of Element 2 Shape Functions" << endl;
      shapes_tangent.Print(cout, dim);
      cout << endl;

      break;
    }
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

