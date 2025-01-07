from setuptools import setup, find_packages

setup(
    name="zonetwelve-LLM-Downsampling",
    version="0.1.0",
    author="ZoneTwelve",
    author_email="admin@zonetwelve.io",
    description="A transformer-based encoder-decoder project for title generation, quantization, and inference.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ZoneTwelve/",
    packages=find_packages(include=["openai_generations_sampling", "src.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "transformers>=4.33.0",
        "optimum[onnxruntime]>=1.9.0",
        "scikit-learn>=1.0",
        "pytest>=7.0.0",
        "fire>=0.5.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache-2.0",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "llm-downsampling=openai_generations_sampling.downsampling:main",
        ]
    },
)