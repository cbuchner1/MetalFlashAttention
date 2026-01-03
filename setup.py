from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

here = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(here, "src")

sources = [
    os.path.join(src_dir, "metal_flash_attention.mm"),
    os.path.join(src_dir, "metal_binding.cpp"),
]

ext_modules = [
    CppExtension(
        name="metal_flash_attn",
        sources=sources,
        include_dirs=[src_dir],
        language="c++",
        extra_compile_args=[
            "-std=c++17",
            "-fPIC",
            "-Wall",
            "-O3",
        ],
        extra_link_args=[
            "-framework", "Metal",
            "-framework", "Foundation",
        ],
    )
]

setup(
    name="metal-flash-attention",
    version="1.0.0",
    description="Metal-accelerated FlashAttention for PyTorch on Apple Silicon",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Metal FlashAttention Contributors",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
    ],
    packages=["metal_flash_attention"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
