from setuptools import setup, find_packages

setup(
    name='ufb',
    version='0.1',
    packages=find_packages(include=['ufb', 'ufb.*']),
    author='UFB Team',
    description='Unary Feedback as Observation: Multi-turn RL for Language Models',
    install_requires=[],  # see requirements.txt
    python_requires='>=3.10',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
    ]
)
