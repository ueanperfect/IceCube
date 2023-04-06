from setuptools import setup

setup(
    name='Model',
    version='1.0.0',
    description='My Awesome App',
    author='Your Name',
    author_email='you@example.com',
    packages=['Model'],
    entry_points={
        'console_scripts': [
            'myapp=myapp.cli:main',
        ],
    },
)
