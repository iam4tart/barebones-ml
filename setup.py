from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='barebones',
    ext_modules=[
        CppExtension(
            name='barebones.libs.octree',
            sources=[
                'src/data-structure/octree/torch_octree.cpp',
                "src/metric/chamfer-distance/torch_chamfer_distance.cpp"
                ],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)