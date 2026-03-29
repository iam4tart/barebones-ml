from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='libtorch_octree_raii',
    ext_modules=[
        CppExtension(
            name='libtorch_octree_raii',
            sources=['torch_octree_raii.cpp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)