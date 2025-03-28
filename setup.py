from setuptools import setup, find_packages

setup(
    name='credit_card_fraud_detection',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'numpy',
        'matplotlib',
        'seaborn'
    ],
)
