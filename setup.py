from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="src",
    packages=find_packages(include=["src"]),
    version="0.1.0",
    description="Computer Vision Models",
    author="Amrit Purshotam",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "bandit==1.7.0",
            "black==20.8b1",
            "flake8==3.9.2",
            "flake8-bandit==2.1.2",
            "flake8-bugbear==21.4.3",
            "flake8-builtins==1.5.3",
            "flake8-comprehensions==3.5.0",
            "isort==5.8.0",
            "mypy==0.902",
            "rope==0.19.0",
            "tensorboard-plugin-profile==2.4.0",
        ],
        "hooks": ["pre-commit==2.11.1"],
        "notebook": [
            "ipywidgets==7.6.3",
            "jupyterlab==3.0.16",
            "widgetsnbextension==3.5.1",
        ],
        "test": ["pytest==6.2.4"],
    },
)
