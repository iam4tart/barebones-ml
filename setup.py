from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='barebones',
    ext_modules=[
        CppExtension(
            name='barebones.libs.octree',
            sources=[
                'src/data-structure/octree/torch_octree.cpp',
            ],
            extra_compile_args=['/view', '/O2'],
            extra_link_args=['/OPT:REF']
        ),
        CppExtension(
            name='barebones.libs.chamfer',
            sources=[
                'src/metric/chamfer-distance/torch_chamfer_distance.cpp',
            ],
            extra_compile_args=['/view', '/O2'],
            extra_link_args=['/OPT:REF']
        ),
        CppExtension(
            name='barebones.libs.fps',
            sources=[
                'src/algorithm/fps/torch_fps.cpp',
            ],
            extra_compile_args=['/view', '/O2'],
            extra_link_args=['/OPT:REF']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)