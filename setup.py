from setuptools import setup, find_packages

setup(
    name="realtime-ai",
    version="0.1.8",
    description="Python SDK for real-time audio processing with OpenAI's Realtime REST API.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jhakulin/realtime-ai",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pyaudio",
        "numpy",
        "websockets",
        "websocket-client",
        "azure-cognitiveservices-speech",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)