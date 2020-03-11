from distutils.core import setup, Extension

module1 = Extension('lfm_C',
                    include_dirs = ['/usr/local/include', '/usr/include/python3.6m',  "/home/chitianqilin/.local/lib/python3.6/site-packages/numpy/core/include/numpy"],
                    library_dirs = ['/usr/local/lib'],
                    sources = ['lfm_C.cc', 'Faddeeva.cc' ])

setup (name = 'lfm_C',
        version = '0.01',
        description = 'This is a package for acceleration of latent force model',
        author = 'Tianqi Wei',
        author_email = 't.wei@sheffield.ac.uk',
        ext_modules = [module1])
