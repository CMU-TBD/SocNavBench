
# This version of pyassimp has a bug which can be fixed by following this commit
# https://github.com/assimp/assimp/commit/b6d3cbcb61f4cc4c42678d5f183351f95c97c8d4

patch $CONDA_PREFIX/lib/python3.6/site-packages/pyassimp/core.py patches/pyassimp_inst.py.patch

