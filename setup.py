from setuptools import setup

setup(
    name='frecog',
    version='0.0.1',
    py_modules=['cli'],
    install_requires=[
        'Click',
        'nose2',
        'face_recognition'
    ],
    author='TeamOne',
    entry='''
        [console_scripts]
        frecog=cli:manager
    '''
)