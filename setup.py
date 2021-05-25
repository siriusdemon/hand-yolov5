import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()


setuptools.setup(
    name = "handdet",
    version = "1.0.0",
    author = "sirius demon",
    author_email = "mory2016@126.com",
    description="handdet using yolov5",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url = "https://github.com/siriusdemon/hand-yolov5",
    packages=setuptools.find_packages(),
    package_data = {
        'handdet': ['checkpoints/best_sd.pt'],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)