from setuptools import setup, find_packages

setup(
    name='cluster_analysis',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'opencv-python',
        'scipy',
    ],
    authors='Grebenyuk Vasilii, Lukiev Ivan',
    authors_email='vasya.31.46@gmail.com, 123@gmail.com',
    description='Library for cluster analysis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/crystal_analysis',        #######
)