"""Build script — installs the .pth site-hook into site-packages ROOT.

A .pth file must land at the site-packages root (purelib) for Python's site.py
to execute it at interpreter start. setuptools `data_files` puts files into the
`data` scheme category (NOT purelib), so a .pth shipped via data_files does NOT
get executed. The reliable fix: a build_py cmdclass that copies the .pth into
build_lib root, whence bdist_wheel packages it into the purelib category.
"""
import os
from setuptools import setup
from setuptools.command.build_py import build_py


class build_py_with_pth(build_py):
    def run(self):
        super().run()
        here = os.path.dirname(os.path.abspath(__file__))
        src = os.path.join(here, "wan_vae_fp16acc.pth")
        dst = os.path.join(self.build_lib, "wan_vae_fp16acc.pth")
        self.copy_file(src, dst)
        # record for the wheel's RECORD + make sure bdist_wheel includes it
        self.outputs = getattr(self, "outputs", [])
        self.outputs.append(dst)


setup(
    cmdclass={"build_py": build_py_with_pth},
)
