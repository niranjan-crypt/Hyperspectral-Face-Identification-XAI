from setuptools import setup, find_packages

setup(
    name="hyperspectral_face_id_xai",
    version="0.1.0",
    description="Real-Time Hyperspectral Human Face Identification Using Spectral-Spatial Feature Fusion and Explainable AI",
    author="Niranjan Krishnakumar, Akshath Senthilkumar, Arjun M, Nikhil R",
    author_email="niranjankrishnakumar2005@gmail.com",
    url="https://github.com/niranajan-crypto/Hyperspectral-Face-Identification-XAI",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "scikit-image",
        "opencv-python",
        "pillow",
        "matplotlib",
        "seaborn",
        "pandas",
        "pyyaml",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
