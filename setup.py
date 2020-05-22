from setuptools import setup

setup(
    name='frecog',
    version='0.0.2',
    py_modules=['main'],
    install_requires=[
        'Click',
        'nose2',
        'configparser',
        'opencv-python',
        'matplotlib',
        'sklearn',
        'cmake',
        'pillow',
        'scipy',
        'keras',
        'h5py',
        'tensorflow',
        'dlib',
        'imutils',
        'keyboard',
        'pandas'
        
    ],
    author='TeamOne',
    entry_points='''
        [console_scripts]
        frecog=cli:frecog
    '''
)