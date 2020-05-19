from setuptools import setup

setup(
    name='frecog',
    version='0.0.1',
    py_modules=['main'],
    install_requires=[
        'Click',
        'nose2',
        'configparser',
        'opencv-python',
        'matplotlib',
        'sklearn',
        'keras',
        'tensorflow',
        'dlib',
        'imutils',
        'keyboard'
        
    ],
    author='TeamOne',
    entry_points='''
        [console_scripts]
        frecog=cli:frecog
    '''
)