//                                MFEM Example 3 - Electromagnetic (Parallel)
//
//
// Sample run:   ./pumi_ex3p -m /users/maliks2/Meshes/MaxwellMeshes/Fichera/coarse/fichera.smb -p /users/maliks2/Meshes/MaxwellMeshes/Fichera/fichera.x_t -o 1 > pumi
//               ./pumi_ex3p -m /users/maliks2/Meshes/MaxwellMeshes/Fichera/25k/fichera_25k.smb -p /users/maliks2/Meshes/MaxwellMeshes/Fichera/25k/fichera_25k_nat.x_t -o 1 > pumi
//
// Description:  The purpose of this example is to execute Maxwell Problem in MFEM and transfer the Electric Field
//               to PUMI field.

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
#include <em.h>

using namespace std;
using namespace mfem;

void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
void pumiUserFunction(const apf::Vector3& x, apf::Vector3& f);
double freq = 1.0, kappa;
int dim;
double alpha = 0.25;  double beta = 2.0;

int main(int argc, char *argv[])
{
    // 0. initilize mpi
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 1. Parse command-line options.
   const char *mesh_file = "../data/pumi/serial/sphere.smb";
#ifdef MFEM_USE_SIMMETRIX
   const char *model_file = "../data/pumi/geom/sphere.x_t";
#else
   const char *model_file = "../data/pumi/geom/sphere.dmg";
#endif
  int order = 1;
  bool static_cond = false;
  bool visualization = 1;
  OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                 "Mesh file to use.");
  args.AddOption(&model_file, "-p", "--parasolid",
                 "Parasolid model to use.");
  args.AddOption(&order, "-o", "--order",
                 "Finite element order (polynomial degree).");
  args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                 "--no-visualization",
                 "Enable or disable GLVis visualization.");
  args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                 "--no-static-condensation", "Enable static condensation.");

  args.Parse();
  if (!args.Good())
  {
    if(myid == 0)
    {
      args.PrintUsage(cout);
    }
    MPI_Finalize();
    return 1;
  }
  if (myid == 0)
  {
     args.PrintOptions(cout);
  }
  kappa = freq * M_PI;

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


  //If it is higher order change shape
  if (order > 1){
      crv::BezierCurver bc(pumi_mesh, order, 2);
      bc.run();
  }
  pumi_mesh->verify();

  // 2. Create the MFEM mesh object from the PUMI mesh. We can handle triangular
  //    and tetrahedral meshes. Other inputs are the same as MFEM default
  //    constructor.
  ParPumiMesh* pmesh = new ParPumiMesh(MPI_COMM_WORLD, pumi_mesh);
  pmesh->ReorientTetMesh(); // *** for ND spaces
  dim = pmesh->Dimension();
  int sdim = pmesh->SpaceDimension();

  // 3. Define a parallel finite element space on the parallel mesh. Here we
  //    use the Nedelec finite elements of the specified order.
  FiniteElementCollection *fec = new ND_FECollection(order, dim);
  ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
  HYPRE_Int size = fespace->GlobalTrueVSize();
  if (myid == 0)
  {
     cout << "Number of finite element unknowns: " << size << endl;
  }

  // 4. Determine the list of true (i.e. parallel conforming) essential
  //    boundary dofs. In this example, the boundary conditions are defined
  //    by marking all the boundary attributes from the mesh as essential
  //    (Dirichlet) and converting them to a list of true dofs.
  Array<int> ess_tdof_list;
  if (pmesh->bdr_attributes.Size())
  {
     Array<int> ess_bdr(pmesh->bdr_attributes.Max());
     ess_bdr = 1;
     fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
  }

  // 5. Set up the parallel linear form b(.) which corresponds to the
  //    right-hand side of the FEM linear system, which in this case is
  //    (f,phi_i) where f is given by the function f_exact and phi_i are the
  //    basis functions in the finite element fespace.
  VectorFunctionCoefficient f(sdim, f_exact);
  ParLinearForm *b = new ParLinearForm(fespace);
  b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
  b->Assemble();

  // 6. Define the solution vector x as a parallel finite element grid function
  //    corresponding to fespace. Initialize x by projecting the exact
  //    solution. Note that only values from the boundary edges will be used
  //    when eliminating the non-homogeneous boundary condition to modify the
  //    r.h.s. vector b.
  ParGridFunction x(fespace);
  VectorFunctionCoefficient E(sdim, E_exact);
  x.ProjectCoefficient(E);

  // 7. Set up the parallel bilinear form corresponding to the EM diffusion
  //     operator curl muinv curl + sigma I, by adding the curl-curl and the
  //     mass domain integrators.
  Coefficient *muinv = new ConstantCoefficient(1.0);
  Coefficient *sigma = new ConstantCoefficient(1.0);
  ParBilinearForm *a = new ParBilinearForm(fespace);
  a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
  a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));

  // 8. Assemble the parallel bilinear form and the corresponding linear
  //     system, applying any necessary transformations such as: parallel
  //     assembly, eliminating boundary conditions, applying conforming
  //     constraints for non-conforming AMR, static condensation, etc.
  if (static_cond) { a->EnableStaticCondensation(); }
  a->Assemble();

  HypreParMatrix A;
  Vector B, X;
  a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

  if (myid == 0)
  {
     cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
  }

  // 9. Define and apply a parallel PCG solver for AX=B with the AMS
  //     preconditioner from hypre.
  ParFiniteElementSpace *prec_fespace =
     (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
  HypreSolver *ams = new HypreAMS(A, prec_fespace);
  HyprePCG *pcg = new HyprePCG(A);
  pcg->SetTol(1e-12);
  pcg->SetMaxIter(500);
  pcg->SetPrintLevel(2);
  pcg->SetPreconditioner(*ams);
  pcg->Mult(B, X);

  // 10. Recover the parallel grid function corresponding to X. This is the
  //     local finite element solution on each processor.
  a->RecoverFEMSolution(X, *b, x);

  // 11. Compute and print the L^2 norm of the error.
  {
     double err = x.ComputeL2Error(E);
     if (myid == 0)
     {
        cout << "\n|| E_h - E ||_{L^2} = " << err << '\n' << endl;
     }
  }

  // 12. Save the refined mesh and the solution in parallel. This output can
  //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
  {
     ostringstream mesh_name, sol_name;
     mesh_name << "mesh." << setfill('0') << setw(6) << myid;
     sol_name << "sol." << setfill('0') << setw(6) << myid;

     ofstream mesh_ofs(mesh_name.str().c_str());
     mesh_ofs.precision(8);
     pmesh->Print(mesh_ofs);

     ofstream sol_ofs(sol_name.str().c_str());
     sol_ofs.precision(8);
     x.Save(sol_ofs);
  }

  // 13. Send the solution by socket to a GLVis server.
  if (visualization)
  {
     char vishost[] = "localhost";
     int  visport   = 19916;
     socketstream sol_sock(vishost, visport);
     sol_sock << "parallel " << num_procs << " " << myid << "\n";
     sol_sock.precision(8);
     sol_sock << "solution\n" << *pmesh << x << flush;
  }

  // 14. Create Nedelec Field on PUMI mesh.
  apf::Field* nedelecField = apf::createField(
              pumi_mesh, "nedelec_field", apf::SCALAR, apf::getNedelec(order));

  // 15. Field Transfer from MFEM to PUMI.
  pmesh->NedelecFieldMFEMtoPUMI(pumi_mesh, &x, nedelecField);


  // 16. Estimate Error - (Equilibration of Residuals Method)
  apf::Field* residualErrorField = em::estimateError(nedelecField);

  cout << "======== Error Field =========" << endl;
  apf::MeshEntity* tet;
  apf::MeshIterator* it = pumi_mesh->begin(3);
  while ((tet = pumi_mesh->iterate(it))) {
    cout << apf::getScalar(residualErrorField, tet, 0) << endl;
  }
  pumi_mesh->end(it);



  // 18. Write Fields to PUMI
  apf::Field* E_exact_Field = 0;
  apf::Field* E_fem_Field = 0;
  apf::Field* E_exact_nodal_Field = 0;
  apf::Field* E_fem_nodal_Field = 0;
  apf::Field* exactErrorField = 0;
  // apf::Field* residualErrorField = 0 (defined above);
  apf::Field* exactErrorNodalField = 0;
  apf::Field* residualErrorNodalField = 0;

  E_exact_Field = apf::createIPField(pumi_mesh, "E_exact_field", apf::VECTOR, 1);
  E_fem_Field = apf::createIPField(pumi_mesh, "E_fem_field", apf::VECTOR, 1);
  E_exact_nodal_Field = apf::createField(pumi_mesh, "E_exact_nodal_field", apf::VECTOR, apf::getLagrange(1));
  E_fem_nodal_Field = apf::createField(pumi_mesh, "E_fem_nodal_field", apf::VECTOR, apf::getLagrange(1));
  exactErrorField = apf::createIPField(pumi_mesh, "exact_error_field", apf::SCALAR, 1);
  exactErrorNodalField = apf::createField(pumi_mesh, "exact_error_nodal_field", apf::SCALAR, apf::getLagrange(1));
  residualErrorNodalField = apf::createField(pumi_mesh, "residual_error_nodal_field", apf::SCALAR, apf::getLagrange(1));

  // 18(a). Write electric field (exact and FEM) to PUMI mesh
  apf::MeshEntity* ent;
  apf::MeshIterator* itr = pumi_mesh->begin(3);
  while ((ent = pumi_mesh->iterate(itr)))  // iterate over all regions
  {
    apf::Vector3 c = apf::Vector3(1./3.,1./3.,1./3.);

    apf::MeshElement* me = apf::createMeshElement(pumi_mesh, ent);
    apf::Element* el = apf::createElement(nedelecField, me);

    // write fem field (Vector)
    apf::Vector3 vvalue;
    apf::getVector(el, c, vvalue);
    apf::setVector(E_fem_Field, ent, 0, vvalue);

    // write exact field (Vector)
    apf::Vector3 cg;
    apf::mapLocalToGlobal(me, c, cg);
    pumiUserFunction(cg, vvalue);
    apf::setVector(E_exact_Field, ent, 0, vvalue);

    apf::destroyElement(el);
    apf::destroyMeshElement(me);
  }
  pumi_mesh->end(itr);
  cout << "Electric Fields Written" << endl;

  // 18(b). Write exact error field // TODO


  // 18(c). Convert Lagrange Constant Fields to Nodal Fields
  itr = pumi_mesh->begin(0);
  while ((ent = pumi_mesh->iterate(itr)))
  {
    apf::Adjacent elements;
    pumi_mesh->getAdjacent(ent, 3, elements);
    int ne = (int) elements.getSize();

    // (i). error nodal fields (exact, residual)
    double exact_errors[ne];
    double residual_errors[ne];
    for (std::size_t i=0; i < ne; ++i){
      //exact_errors[i] = apf::getScalar(exactErrorField, elements[i], 0);
      residual_errors[i] = apf::getScalar(residualErrorField, elements[i], 0);
    }

    double exact_average = 0.0;
    double residual_average = 0.0;
    for(int i = 0; i < ne; i++){
      //exact_average += exact_errors[i];
      residual_average += residual_errors[i];
    }
    //exact_average = exact_average/ne;
    residual_average = residual_average/ne;

    //apf::setScalar(exactErrorNodalField, ent, 0, exact_average);
    apf::setScalar(residualErrorNodalField, ent, 0, residual_average);

    // (ii). solution nodal fields (exact, FEM)
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
  cout << "Error Fields Written" << endl;
  



  apf::writeVtkFiles("error_Fields_pumi", pumi_mesh);

  // 17. Perform Mesh Adapt

  delete pcg;
  delete ams;
  delete a;
  delete sigma;
  delete muinv;
  delete b;
  delete fespace;
  delete fec;
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
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(2));
      E(2) = sin(kappa * x(0));
   }
   else
   {
      E(0) = sin(kappa * x(1));
      E(1) = sin(kappa * x(0));
      if (x.Size() == 3) { E(2) = 0.0; }
   }
}

void f_exact(const Vector &x, Vector &f)
{
   if (dim == 3)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
      if (x.Size() == 3) { f(2) = 0.0; }
   }
}

// TODO
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

/*
double computeElementExactError(apf::Mesh* mesh, apf::MeshEntity* e,
  apf::Field* f, mth::Vector<double> const error_dofs)                              
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
  int order = 2 * fs->getOrder();                                                   
  int np = apf::countIntPoints(me, order);                                          
                                                                                    
  apf::NewArray<apf::Vector3> vectorshape(nd);                                      
                                                                                    
  apf::Vector3 p;                                                                   
  for (int i = 0; i < np; i++) {                                                    
    apf::getIntPoint(me, order, i, p);                                              
    double weight = apf::getIntWeight(me, order, i);                                
    apf::Matrix3x3 J;                                                               
    apf::getJacobian(me, p, J);                                                     
    double jdet = apf::getJacobianDeterminant(J, dim);                              
    w = weight * jdet;                                                              
                                                                                    
    apf::getVectorShapeValues(el, p, vectorshape);                                  
    // TODO take care of negative dof values                                        
    mth::Matrix<double> vectorShape (nd, dim);                                      
    for (int j = 0; j < nd; j++)                                                    
      for (int k = 0; k < dim; k++)                                                 
        vectorShape(j,k) = vectorshape[j][k];                                       
                                                                                    
    mth::Matrix<double> vectorShapeT (dim, nd);                                     
    mth::transpose(vectorShape, vectorShapeT);                                      
                                                                                    
    mth::Vector<double> err_func;                                                   
    mth::multiply(vectorShapeT, error_dofs, err_func);                              
                                                                                    
    error += w * (err_func * err_func);                                             
  }                                                                                 
  if (error < 0.0)                                                                  
    error = -error;                                                                 
                                                                                    
  return sqrt(error);                                                               
}



double computeExactElementError(int elemNo, VectorCoefficient &exsol, Mesh* mesh,
                                  FiniteElementSpace* fespace, GridFunction x)
{
  double error = 0.0;
  const FiniteElement *fe = fespace->GetFE(elemNo);
  int intorder = 2*fe->GetOrder() + 1;
  ElementTransformation* eltr = fespace->GetElementTransformation(elemNo);
  const IntegrationRule *ir = &IntRules.Get(fe->GetGeomType(), intorder);
  DenseMatrix vals, exact_vals;
  Vector loc_errs;

  x.GetVectorValues(*eltr, *ir, vals);
  exsol.Eval(exact_vals, *eltr, *ir);
  vals -= exact_vals;
  loc_errs.SetSize(vals.Width());
  vals.Norm2(loc_errs); 
 
  for (int j = 0; j < ir->GetNPoints(); j++)
  {
    const IntegrationPoint &ip = ir->IntPoint(j);
    eltr->SetIntPoint(&ip);
    error += ip.weight * eltr->Weight() * (loc_errs(j) * loc_errs(j));
  }
  if (error < 0.0)
  {
    return -sqrt(-error);
  }
  return sqrt(error);
}*/

