# Script to create virtual environment with Anaconda and install needed packages
conda create --name cxr python=3.9 -y 
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cxr
conda install pip -y
pip install -r requirements.txt