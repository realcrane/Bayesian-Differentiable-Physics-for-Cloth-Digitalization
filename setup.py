import glob
from setuptools import setup, Extension
from torch.utils import cpp_extension

ALGLIB_DIR = '/usr/not-backed-up/scdg/ProLibs/alglib/src/'

INCLUDE_DIR = cpp_extension.include_paths()
INCLUDE_DIR.append('./')
INCLUDE_DIR.append(ALGLIB_DIR)
INCLUDE_DIR.append('/usr/not-backed-up/scdg/ProLibs/boost_1_75_0/')
INCLUDE_DIR.append('/usr/not-backed-up/scdg/ProLibs/eigen-3.3.9/')

SOURCES = ['bind.cpp', 'scr/Mesh.cpp', 'scr/MeshIO.cpp', 'scr/Geometry.cpp',
		'scr/DDE.cpp', 'scr/Handle.cpp', 'scr/Constraint.cpp', 'scr/Solve.cpp', 'scr/Physics.cpp', 
		'scr/Cloth.cpp', 'scr/Obstacle.cpp', 'scr/Simulation.cpp', 'scr/Collision.cpp', 
		'scr/CollisionUtil.cpp', 'scr/Auglag.cpp', 'scr/BVH.cpp', 'scr/Proximity.cpp']

SOURCES = SOURCES + glob.glob(ALGLIB_DIR + "*.cpp")

setup(name='diffsimMulGPU',
	install_requires=['torch'],
	ext_modules=[cpp_extension.CppExtension(
		name='diffsimMulGPU', 
		include_dirs= INCLUDE_DIR,
		sources= SOURCES,
		extra_compile_args=["-fopenmp"])],
	cmdclass={'build_ext': cpp_extension.BuildExtension})