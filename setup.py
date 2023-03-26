from setuptools import setup, find_packages

setup(
  name = 'local-attention-tf',
  packages = find_packages(),
  version = '0.0.3',
  license='MIT',
  description = 'Local attention, window with lookback, for Image, Audio and language modeling',
  long_description_content_type = 'text/markdown',
  author = 'Muhammad Anas Raza',
  author_email = 'memanasraza@gmail.com',
  url = 'https://github.com/anas-rz/local-attention-tf',
  keywords = [
    'transformers',
    'attention',
    'artificial intelligence'
  ],
  install_requires=[
    'einops>=0.6.0',
    'tensorflow'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)