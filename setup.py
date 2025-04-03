from setuptools import setup

setup(
    name="O2TDM",
    py_modules=["improved_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)