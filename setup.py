from setuptools import setup, find_packages

setup(
    name="verbal-confidence",
    version="0.1.0",
    description="Mechanistic interpretability of LLM verbal confidence expressions",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.29.0",
        "datasets>=2.19.0",
        "huggingface_hub>=0.22.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "h5py>=3.11.0",
    ],
)
