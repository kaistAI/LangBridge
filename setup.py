from setuptools import setup, find_packages

setup(
    name='langbridge',
    packages=find_packages(exclude=[]),
    version='0.0.1',
    license='MIT',
    description='LANGBRIDGE: Multilingual Reasoning Without Multilingual Supervision',
    author='Dongkeun Yoon',
    url='https://github.com/kaistAI/LangBridge',
    python_requires='>=3.7',
    long_description_content_type='text/markdown',
    keywords=[
        'artificial intelligence',
        'deep learning',
        'transformers',
        'natural language processing',
        'large language models'
    ],
    # datasets
    # deepspeed
    # einops
    # einops-exts
    # huggingface-hub
    # pandas
    # pytorch-lightning
    # sentencepiece
    # torchinfo
    # transformers
    # wandb
    install_requires=[
        'datasets',
        'deepspeed',
        'einops',
        'einops-exts',
        'huggingface-hub',
        'pandas',
        'pytorch-lightning',
        'sentencepiece',
        'torchinfo',
        'transformers',
        'wandb',
    ],
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)
