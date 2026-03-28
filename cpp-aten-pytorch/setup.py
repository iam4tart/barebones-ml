from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="c_double_tensor",
    ext_modules=[
        CppExtension(
            'c_double_tensor',
            ['c_double_tensor.cpp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

# cmd -> python setup.py build_ext --inplace && rmdir /s /q build
# powershell -> python setup.py build_ext --inplace ;  Remove-Item -Recurse -Force build