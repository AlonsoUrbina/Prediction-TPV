from setuptools import setup, find_packages

setup(
    name="tpv-prediction",
    version="1.0.0",
    description="Sistema de predicción de TPV para PSPs usando modelos de machine learning",
    author="Alonso Urbina",
    author_email="alonso.urbina@ug.uchile.cl",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        # Gradient Boosting
        "lightgbm>=4.0.0",
        "catboost>=1.2",
        "xgboost>=2.0.0",
        # Hiperparámetros y optimización
        "optuna>=3.3.0",
        # Utilities
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyarrow>=12.0.0",
        "joblib>=1.3.0",
        "holidays>=0.30",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)