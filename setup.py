from setuptools import setup, find_packages

setup(
    name="fils-framework",
    version="0.1.0",
    author="Fabrice Fils-Aimé",
    author_email="",
    description="Adaptive analogical learning for AI, ML, Python and statistics",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fabthebest/fils-framework",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "fils": ["data/**/*.md"],
    },
    python_requires=">=3.8",
    install_requires=[
        "pyyaml",
    ],
    extras_require={
        "notebooks": ["jupyter", "pandas", "numpy", "matplotlib", "seaborn"],
        "ml": ["scikit-learn"],
        "dl": ["torch", "transformers"],
        "all": [
            "jupyter", "pandas", "numpy", "matplotlib", "seaborn",
            "scikit-learn", "torch", "transformers",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="ai ml education pedagogy analogies learning",
)
