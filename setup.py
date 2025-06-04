# setup.py is what turns your messy Python scripts into a professional-grade, versioned, portable package that can be reused, deployed, or shipped.
from setuptools import setup, find_packages

setup(
    name="HousingPriceModel",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "scikit-learn",
        "mlflow",
        "joblib"
    ],
    entry_points={
        "console_scripts": [
            "predict=predict:main"
        ]
    }
)
