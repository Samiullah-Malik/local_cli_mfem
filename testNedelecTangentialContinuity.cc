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


#include <mthQR.h>
#include <mth.h>
#include <mth_def.h>

using namespace std;
using namespace mfem;

apf::MeshEntity* getTetOppVert(
    apf::Mesh* m, apf::MeshEntity* t, apf::MeshEntity* f);
apf::Vector3 computeFaceOutwardNormal(apf::Mesh* m,
    apf::MeshEntity* t, apf::MeshEntity* f, apf::Vector3 const& p);
apf::Vector3 computeFaceNormal(apf::Mesh* m,
    apf::MeshEntity* f, apf::Vector3 const& p);

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
  cout << "MFEM MESH" << endl;
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
      cout << "=================================================" << endl;

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
      cout << "=================================================" << endl;
      cout << "=================================================" << endl;
      cout << endl;

      break;
    }
  }

  // 5. Create Nedelec Field on PUMI mesh.
  apf::Field* nedelecField = apf::createField(
              pumi_mesh, "nedelec_field", apf::SCALAR, apf::getNedelec(order));
  apf::zeroField(nedelecField);
  apf::FieldShape* nedelecFieldShape = nedelecField->getShape();

  // 6. Test Tangential Continuity of Nedelec Vector Shape Functions on PUMI
  //    mesh.
  cout << "PUMI MESH" << endl;
  apf::MeshEntity* face;
  apf::MeshIterator* it;
  it = pumi_mesh->begin(2);
  while ( face = pumi_mesh->iterate(it) ) {

    apf::Up up;
    pumi_mesh->getUp(face, up);
    if (up.n == 2) {

      // elem1
      apf::MeshEntity* tet1 = up.e[0];
      apf::Vector3 xi = apf::Vector3(1./3., 1./3., 0.);

      apf::MeshElement* me1 = apf::createMeshElement(pumi_mesh, tet1);
      apf::Element* el1 = apf::createElement(nedelecField, me1);

      int type = pumi_mesh->getType(tet1);
      int nd = apf::countElementNodes(nedelecFieldShape, type);
      int dim = apf::getDimension(pumi_mesh, tet1);
      apf::NewArray<apf::Vector3> vectorshapes1 (nd);

      apf::Vector3 tet1xi = apf::boundaryToElementXi(pumi_mesh, face, tet1, xi);
      apf::getVectorShapeValues(el1, tet1xi, vectorshapes1);
      mth::Matrix<double> vectorShapes1 (nd, dim);
      for (int j = 0; j < nd; j++)
        for (int k = 0; k < dim; k++)
          vectorShapes1(j,k) = vectorshapes1[j][k];

      apf::destroyElement(el1);
      apf::destroyMeshElement(me1);

      apf::Vector3 n1 = computeFaceOutwardNormal(pumi_mesh, tet1, face, xi);
      mth::Vector<double> nor1 (3);
      nor1(0) = n1[0]; nor1(1) = n1[1]; nor1(2) = n1[2];

      apf::Downward edges;
      int ne = pumi_mesh->getDownward(tet1, 1, edges);
      int which, rotate; bool flip;
      for (int i = 0; i < ne; i++) {
        apf::getAlignment(pumi_mesh, tet1, edges[i], which, flip, rotate);
        if (flip) {
          for(int j = 0; j < dim; j++)
            vectorShapes1(i,j) = -1*vectorShapes1(i,j);
        }
      }

      // compute tangential components of element shape functions
      mth::Vector<double> shapes_mult_normal;
      mth::multiply(vectorShapes1, nor1, shapes_mult_normal);

      mth::Matrix<double> shapes_normal (nd, dim);
      shapes_normal.zero();

      for (int i = 0; i < nd; i++) {
        mth::Vector<double> shape_nc (dim);
        shape_nc[0] = nor1[0]; shape_nc[1] = nor1[1]; shape_nc[2] = nor1[2];

        shape_nc *= shapes_mult_normal(i);
        shapes_normal(i, 0) = shape_nc[0];
        shapes_normal(i, 1) = shape_nc[1];
        shapes_normal(i, 2) = shape_nc[2];
      }

      mth::Matrix<double> shapes_tangent (nd, dim);
      shapes_tangent.zero();
      shapes_tangent += vectorShapes1;
      shapes_tangent -= shapes_normal;

      cout <<"Tangential Components of Element 1 Shape Functions" << endl;
      cout << shapes_tangent << endl;
      cout << "=================================================" << endl;
      cout << endl;

      // elem2
      apf::MeshEntity* tet2 = up.e[1];

      apf::MeshElement* me2 = apf::createMeshElement(pumi_mesh, tet2);
      apf::Element* el2 = apf::createElement(nedelecField, me2);

      type = pumi_mesh->getType(tet2);
      nd = apf::countElementNodes(nedelecFieldShape, type);
      apf::NewArray<apf::Vector3> vectorshapes2 (nd);

      apf::Vector3 tet2xi = apf::boundaryToElementXi(pumi_mesh, face, tet2, xi);
      apf::getVectorShapeValues(el2, tet2xi, vectorshapes2);
      mth::Matrix<double> vectorShapes2 (nd, dim);
      for (int j = 0; j < nd; j++)
        for (int k = 0; k < dim; k++)
          vectorShapes2(j,k) = vectorshapes2[j][k];

      apf::destroyElement(el2);
      apf::destroyMeshElement(me2);

      apf::Vector3 n2 = computeFaceOutwardNormal(pumi_mesh, tet2, face, xi);
      mth::Vector<double> nor2 (3);
      nor2(0) = n2[0]; nor2(1) = n2[1]; nor2(2) = n2[2];


      ne = pumi_mesh->getDownward(tet2, 1, edges);
      for (int i = 0; i < ne; i++) {
        apf::getAlignment(pumi_mesh, tet2, edges[i], which, flip, rotate);
        if (flip) {
          for(int j = 0; j < dim; j++)
            vectorShapes2(i,j) = -1*vectorShapes2(i,j);
        }
      }

      // compute tangential components of element shape functions
      mth::multiply(vectorShapes2, nor2, shapes_mult_normal);

      shapes_normal.zero();

      for (int i = 0; i < nd; i++) {
        mth::Vector<double> shape_nc (dim);
        shape_nc[0] = nor2[0]; shape_nc[1] = nor2[1]; shape_nc[2] = nor2[2];

        shape_nc *= shapes_mult_normal(i);
        shapes_normal(i, 0) = shape_nc[0];
        shapes_normal(i, 1) = shape_nc[1];
        shapes_normal(i, 2) = shape_nc[2];
      }

      shapes_tangent.zero();
      shapes_tangent += vectorShapes2;
      shapes_tangent -= shapes_normal;

      cout <<"Tangential Components of Element 2 Shape Functions" << endl;
      cout << shapes_tangent << endl;
      cout << "=================================================" << endl;
      cout << endl;

      break;
    }
  }
  pumi_mesh->end(it);




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

apf::MeshEntity* getTetOppVert(
    apf::Mesh* m, apf::MeshEntity* t, apf::MeshEntity* f)
{
  apf::Downward fvs;
  int fnv = m->getDownward(f, 0, fvs);
  apf::Downward tvs;
  int tnv = m->getDownward(t, 0, tvs);
  PCU_ALWAYS_ASSERT(tnv == 4 && fnv == 3);
  for (int i = 0; i < tnv; i++) {
    if (apf::findIn(fvs, fnv, tvs[i]) == -1)
      return tvs[i];
  }
  return 0;
}

apf::Vector3 computeFaceNormal(apf::Mesh* m,
    apf::MeshEntity* f, apf::Vector3 const& p)
{
  // Compute face normal using face Jacobian
  apf::MeshElement* me = apf::createMeshElement(m, f);
  apf::Matrix3x3 J;
  apf::getJacobian(me, p, J);
  apf::destroyMeshElement(me);

  apf::Vector3 g1 = J[0];
  apf::Vector3 g2 = J[1];
  apf::Vector3 n = apf::cross( g1, g2 );
  return n.normalize();
}

apf::Vector3 computeFaceOutwardNormal(apf::Mesh* m,
    apf::MeshEntity* t, apf::MeshEntity* f, apf::Vector3 const& p)
{
  apf::Vector3 n = computeFaceNormal(m, f, p);

  // Orient the normal outwards from the tet
  apf::MeshEntity* oppVert = getTetOppVert(m, t, f);
  apf::Vector3 vxi = apf::Vector3(0.,0.,0.);

  // get global coordinates of the vertex
  apf::Vector3 txi;
  m->getPoint(oppVert, 0, txi);

  // get global coordinates of the point on the face
  apf::MeshElement* fme = apf::createMeshElement(m, f);
  apf::Vector3 pxi;
  apf::mapLocalToGlobal(fme, p, pxi);
  apf::destroyMeshElement(fme);

  apf::Vector3 pxiTotxi = txi - pxi;
  //std::cout << "dot product " << pxiTotxi*n << std::endl;
  if (pxiTotxi*n > 0) {
    n = n*-1.;
  }
  return n;
}
