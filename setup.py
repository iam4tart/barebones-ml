import os
import shutil
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='barebones',
    packages=['barebones', 'barebones.libs'],
    ext_modules=[
        CppExtension(
            name='barebones.libs.octree',
            sources=['src/data-structure/octree/torch_octree.cpp'],
            # /W3 = warning level 3, /O2 = optimize, /EHsc = exception handling
            # NOT add /view — not a valid msvc flag
            extra_compile_args=['/W3', '/O2', '/EHsc'],
            extra_link_args=['/OPT:REF'],
        ),
        CppExtension(
            name='barebones.libs.chamfer',
            sources=['src/metric/chamfer-distance/torch_chamfer_distance.cpp'],
            extra_compile_args=['/W3', '/O2', '/EHsc'],
            extra_link_args=['/OPT:REF'],
        ),
        CppExtension(
            name='barebones.libs.fps',
            sources=['src/algorithm/fps/torch_fps.cpp'],
            extra_compile_args=['/W3', '/O2', '/EHsc'],
            extra_link_args=['/OPT:REF'],
        ),
        CppExtension(
            name='barebones.libs.ball_query',
            sources=['src/algorithm/ball-query/torch_ball_query.cpp'],
            extra_compile_args=['/W3', '/O2', '/EHsc'],
            extra_link_args=['/OPT:REF'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)