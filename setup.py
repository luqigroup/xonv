import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

reqs = []
setuptools.setup(
    name='extconv',
    version='0.1',
    author='Ali Siahkoohi',
    author_email='alisk@rice.edu',
    description='Spatially varying convolutional layers for PyTorch.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/alisiahkoohi/extconv',
    license='MIT',
    install_requires=reqs,
    packages=setuptools.find_packages(),
)
