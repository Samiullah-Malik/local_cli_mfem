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

using namespace std;
using namespace mfem;

void E_exact(const Vector &, Vector &); // User defined functions
void A_const(const Vector &, Vector &);
void A_linear(const Vector &x, Vector &f);
void A_quadratic(const Vector &x, Vector &f);
void A_cubic(const Vector &x, Vector &f);

void NedelecFieldMFEMtoPUMI(mfem::ParPumiMesh* mfem_mesh, apf::Mesh2* apf_mesh, mfem::ParGridFunction* gf, apf::Field* nedelecField);
int findIndex(mfem::Array<int> array, int value);
bool same(int a[], int b[], int size);

void printDenseMatrix(DenseMatrix mat); // (testing function)

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

    elemNo++;
  }


  // 7. Project MFEM Field onto PUMI Field and store dofs
  NedelecFieldMFEMtoPUMI(pmesh, pumi_mesh, &gf, nedelecField);

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
//===================================================================//
//===================================================================//
//===================================================================//


// TESTING METHOD - This method will print a matrix of type Densmatrix
void printDenseMatrix(DenseMatrix mat)
{
  for(int r = 0; r < mat.Height(); r++)
  {
    for(int c = 0; c < mat.Width(); c++)
    {
      cout << mat(r,c) << " ";
    }
    cout << endl;
  }
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

void NedelecFieldMFEMtoPUMI(mfem::ParPumiMesh* mfem_mesh, apf::Mesh2* apf_mesh, mfem::ParGridFunction* gf, apf::Field* nedelecField)
{
  // 1. Check if Numbering exists TODO cleanup    check if numbering is local_vtx_numbering
  apf::Numbering* local_vtx_numbering = apf_mesh->getNumbering(0);
  const char* name = apf::getName(local_vtx_numbering);
  cout << name << endl;
  cout << "Numberings " << apf_mesh->countNumberings() << endl;

  apf::FieldShape* nedelecFieldShape = nedelecField->getShape();
  int num_nodes = 4 * nedelecFieldShape->countNodesOn(0) + // Vertex
                  6 * nedelecFieldShape->countNodesOn(1) + // Edge
                  4 * nedelecFieldShape->countNodesOn(2) + // Triangle
                      nedelecFieldShape->countNodesOn(4);  // Tetrahedron
  int dim = apf_mesh->getDimension();
  apf::NewArray<apf::Vector3> pumi_nodes (num_nodes);

  size_t elemNo = 0;
  apf::MeshEntity* el_ent;
  apf::MeshIterator* el_it;
  el_it = apf_mesh->begin(dim);
  while ( el_ent = apf_mesh->iterate(el_it) ) { // loop over elements

    // collect pumi nodes
    int node_number = 0;
    for (int d = 0; d <= dim; d++) {
      if (nedelecFieldShape->hasNodesIn(d)) {
        apf::Downward a;
        int na = apf_mesh->getDownward(el_ent,d,a);
        for (int i = 0; i < na; i++) {  // loop over downward entities
          int type = apf_mesh->getType(a[i]);
          int nan = nedelecFieldShape->countNodesOn(type);
          for (int n = 0; n < nan; n++) {   // loop over entity nodes
            apf::Vector3 ent_xi;
            nedelecFieldShape->getNodeXi(type, n, ent_xi); // getNodeXi
            apf::Vector3 elem_xi = apf::boundaryToElementXi(
                            apf_mesh, a[i], el_ent, ent_xi); // transform entity nodeXi to parent element nodeXi.
            pumi_nodes[node_number++] = elem_xi;
          }
        }
      }
    }

    // get downward vertices of PUMI element
    apf::Downward v;
    int nv = apf_mesh->getDownward(el_ent,0,v);
    std::vector<int> pumi_vtx_indices (nv);
    for (int i = 0; i < nv; i++)
      pumi_vtx_indices[i] = apf::getNumber(local_vtx_numbering, v[i], 0, 0);

    // get downward vertices of MFEM element
    mfem::Array<int> mfem_vtx_indices;
    mfem_mesh->GetElementVertices(elemNo, mfem_vtx_indices);

    // get rotated indices of PUMI element
    int pumi_tetv[nv];
    for (int i = 0; i < nv; i++)
      pumi_tetv[i] = findIndex(mfem_vtx_indices, pumi_vtx_indices[i]);

    // all possible tet rotations
    int tet_rotation[24][4] =
    {{0,1,2,3} // 0
    ,{0,2,3,1} // 1
    ,{0,3,1,2} // 2
    ,{1,0,3,2} // 3
    ,{1,3,2,0} // 4
    ,{1,2,0,3} // 5
    ,{2,0,1,3} // 6
    ,{2,1,3,0} // 7
    ,{2,3,0,1} // 8
    ,{3,0,2,1} // 9
    ,{3,2,1,0} // 10
    ,{3,1,0,2} // 11

    ,{0,1,3,2} // 12
    ,{0,2,1,3} // 13
    ,{0,3,2,1} // 14
    ,{1,0,2,3} // 15
    ,{1,2,3,0} // 16
    ,{1,3,0,2} // 17
    ,{2,0,3,1} // 18
    ,{2,1,0,3} // 19
    ,{2,3,1,0} // 20
    ,{3,0,1,2} // 21
    ,{3,1,2,0} // 22
    ,{3,2,0,1} // 23
    };

    // get rotation of the mfem rotated tet from the table above
    int rotation;
    for (int i = 0; i < 24; i++) {
      if( same(pumi_tetv,tet_rotation[i],nv) ) {
        rotation = i;
        break;
      }
    }

    // map the coordinates computed on the original set of vertices
    // to the coordinates computed based on a rotated set of vertices
    IntegrationRule mfem_nodes (num_nodes);

    for(int i = 0; i < num_nodes; i++) {
      apf::Vector3 xi = pumi_nodes[i]; // original xi

      double b[4];
      b[0] = 1-xi[0]-xi[1]-xi[2]; b[1] = xi[0]; b[2] = xi[1]; b[3] = xi[2];
      int const* originalIndexOf = tet_rotation[rotation];
      double a[4];
      for (int i = 0; i < 4; i++)
        a[ originalIndexOf[i] ] = b[i];
      xi[0] = a[1]; xi[1] = a[2]; xi[2] = a[3]; // rotated xi

      IntegrationPoint& ip = mfem_nodes.IntPoint(i);
      double pt_crd[3] = { xi[0], xi[1], xi[2] };
      ip.Set(pt_crd,3); // set integration point in mfem
    }

    // evaluate the vector field on the mfem nodes
    ElementTransformation* eltr = mfem_mesh->GetElementTransformation(elemNo);
    DenseMatrix mfem_field_vals;
    gf->GetVectorValues(*eltr, mfem_nodes, mfem_field_vals);

    // compute and store dofs on ND field
    node_number = 0;
    for (int d = 0; d <= dim; d++) {
      if (nedelecFieldShape->hasNodesIn(d)) {
        apf::Downward a;
        int na = apf_mesh->getDownward(el_ent,d,a);
        for (int i = 0; i < na; i++) {  // loop over downward entities
          int type = apf_mesh->getType(a[i]);
          int nan = nedelecFieldShape->countNodesOn(type);
          for (int n = 0; n < nan; n++) {   // loop over entity nodes
            apf::Vector3 xi, tangent;
            nedelecFieldShape->getNodeXi(type, n, xi); // getNodeXi
            nedelecFieldShape->getNodeTangent(type, n, tangent); // getNodeTangent

            apf::Vector3 pumi_field_vector; // getVectorValue in PUMI
            pumi_field_vector[0] = mfem_field_vals(0,node_number);
            pumi_field_vector[1] = mfem_field_vals(1,node_number);
            pumi_field_vector[2] = mfem_field_vals(2,node_number);

            apf::MeshElement* me = apf::createMeshElement(apf_mesh, a[i]);
            apf::Matrix3x3 J; // get Jacobian
            apf::getJacobian(me, xi, J);

            apf::Vector3 temp = J * pumi_field_vector; // compute scalar dof
            double dof = temp * tangent;
            apf::setScalar(nedelecField, a[i], n, dof);

            node_number++;
          }
        }
      }
    }
    elemNo++;
  }
  apf_mesh->end(el_it); // end loop over all elements
}

int findIndex(mfem::Array<int> array, int value)
{
  int size = array.Size();
  for (int i = 0; i < size; i++) {
    if (value == array[i] ) return i;
  }
  return -1;
}

bool same(int a[], int b[], int size)
{
  for (int i = 0; i < size; i++)
    if(a[i] != b[i])
      return false;
  return true;
}

/*void rotateTetXi(apf::Vector3& xi, int rotation)
{
  double b[4];
  b[0] = 1-xi[0]-xi[1]-xi[2]; b[1] = xi[0]; b[2] = xi[1]; b[3] = xi[2];
  int const* originalIndexOf = tet_rotation[rotation];
  double a[4];
  for (int i = 0; i < 4; i++)
    a[ originalIndexOf[i] ] = b[i];
  xi[0] = a[1]; xi[1] = a[2]; xi[2] = a[3];
}*/

