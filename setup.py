from setuptools import setup, find_packages

setup(
    name='dynamicdl',
    version='0.1.1-alpha',
    packages=find_packages(),
    license='Apache License 2.0',
    author='Anthony Tong',
    author_email='atong28.usa@gmail.com',
    description='A PyTorch-based dynamic dataloading library',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/ESB-AI-Lab/DynamicDL',
    install_requires=[
        "opencv-python>=4.6.0,<=4.9.0.80",
        "Pillow>=8.0.0,<=10.3.0",
        "PyYAML>=5.4.0,<=6.0.1",
        "tqdm>=4.1.0,<=4.66.2",
        "numpy>=1.18.0,<2.0.0",
        "pandas>=1.0.0,<=2.2.2",
        "torch>=2.0.0,<=2.3.0",
        "torchvision>=0.15.1,<=0.18.0",
        "typing-extensions>=4.0.0,<=4.11.0",
        "xmltodict>=0.12.0,<=0.13.0",
        "jsonpickle>=3.0.0,<=3.1.0",
        "matplotlib>=3.6.0,<=3.8.4"
    ],
    include_package_data=True
)