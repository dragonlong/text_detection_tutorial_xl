#!/usr/bin/env bash
cd ~
###################################
##		VENV for Python 2		 ##
###################################
virtualenv -p python venv2
source ~/venv2/bin/activate                  # Activate the virtual environment
pip install numpy scipy matplotlib ipython jupyter pandas sympy nose
pip install -U scikit-learn
pip install -U nltk
pip install pillow
pip install h5py
pip install kaggle-cli
pip install scikit-image

pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl
pip install torchvision
python -m pip install pymongo
pip install tqdm
pip install Click
pip install --upgrade tensorflow-gpu
# Install keras from source
git clone https://github.com/fchollet/keras.git
cd keras
python setup.py install
cd ..
rm -rf keras


# Download OpenCV source and install
cd ~/opencv-3.3.0/
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D INSTALL_PYTHON_EXAMPLES=ON \
      -D INSTALL_C_EXAMPLES=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.3.0/modules \
      -D PYTHON_EXECUTABLE=~/venv2/bin/python \
      -D BUILD_EXAMPLES=ON ..
make -j16
sudo make install
sudo ldconfig
cd ~/venv2/lib/python2.7/site-packages/
ln -s /usr/local/lib/python2.7/site-packages/cv2.so cv2.so
deactivate

