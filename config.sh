export PUMI_INSTALL_PATH="/lore/maliks2/redhat-mfem-develop/core-install/"
export MFEM_INSTALL_PATH="/lore/maliks2/redhat-mfem-develop/mfem-build/install"
flags="-g -O0"
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DSCOREC_PREFIX=$PUMI_INSTALL_PATH \
  -DMFEM_PREFIX=$MFEM_INSTALL_PATH \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF \
  -DCMAKE_INSTALL_PREFIX:PATH="$CMAKE_BINARY_DIR/../bin" \
  -DCMAKE_C_COMPILER="mpicc" \
  -DCMAKE_CXX_COMPILER="mpicxx" \
  -DCMAKE_C_FLAGS="${flags}" \
  -DCMAKE_CXX_FLAGS="${flags}" \
  -DCMAKE_EXE_LINKER_FLAGS="-lpthread ${flags}" \
  -DMDS_ID_TYPE="int" \
  -DENABLE_ZOLTAN=ON \
  -DPCU_COMPRESS=ON \
  -DSIM_MPI=mpich3.3 \
  -DSIM_PARASOLID=ON
