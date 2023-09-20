# First-snake-game

#Need to have these installed or to create environment

conda create --name cudaenv
conda activate cudaenv 
conda install cudatoolkit 
pip install dalle2_pytorch 
pip uninstall torch 
pip uninstall torchvision 
pip uninstall torchaudio 
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install -U matplotlib 
pip install pytorch-lightning 
pip install opencv-python 
pip install -U albumentations 
pip install Cython 
pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
