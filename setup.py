from setuptools import setup, find_packages


with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='graphrole',
    version='1.0.0',
    author='Daniel Kaslovsky',
    author_email='dkaslovsky@gmail.com',
    license='MIT',
    description='Automatic feature extraction and node role assignment for transfer learning on graphs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=requirements,
    url='https://github.com/dkaslovsky/GraphRole',
    keywords=[
        'graph', 'feature extraction', 'transfer learning', 'network',
        'graph analysis', 'network analysis', 'refex', 'rolx',
    ],
    classifiers=[
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)
