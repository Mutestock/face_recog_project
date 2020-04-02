from setuptools import setup

setup(
    name='frecog',
    version='0.0.1',
    py_modules=['main'],
    install_requires=[
        'Click',
        'nose2',
        'face_recognition'
    ],
    author='TeamOne',
    entry_points='''
        [console_scripts]
        frecog=cli:frecog
    '''
)