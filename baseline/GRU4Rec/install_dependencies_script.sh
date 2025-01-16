#!/bin/zsh

echo "Start"
set -x


conda create -n gru4rec_conda_pytorch_env python=3.8 -y
conda init
conda activate gru4rec_conda_pytorch_env

conda config --add channels pytorch 
conda config --add channels defaults 
conda config --add channels conda-forge 

conda install asttokens=2.0.5=pyhd3eb1b0_0 -y
conda install backcall=0.2.0=pyhd3eb1b0_0 -y
conda install -c defaults  blas=1.0=openblas -y
conda install ca-certificates=2023.01.10 -y
conda install certifi=2022.12.7 -y
conda install debugpy=1.5.1 -y
conda install decorator=5.1.1=pyhd3eb1b0_0 -y
conda install executing=0.8.3=pyhd3eb1b0_0 -y
conda install ipykernel=6.15  -y
conda install ipython=8.4.0 -y
conda install jedi=0.18.1 -y
conda install jupyter_client=7.0.6=pyhd8ed1ab_0 -y
conda install jupyter_core=4.12.0 -y
conda install libffi=3.3 -y
conda install matplotlib-inline=0.1.6 -y
conda isntall nest-asyncio=1.5.6=pyhd8ed1ab_0 -y
conda install openssl=1.1.1t -y
conda install packaging=23.1=pyhd8ed1ab_0 -y
conda install parso=0.8.3=pyhd3eb1b0_0 -y
conda install pexpect=4.8.0=pyhd3eb1b0_3 -y
conda install pickleshare=0.7.5=pyhd3eb1b0_1003 -y
conda install prompt-toolkit=3.0.20=pyhd3eb1b0_0 -y
conda install psutil=5.9.0 -y
conda install ptyprocess=0.7.0=pyhd3eb1b0_2 -y
conda install pure_eval=0.2.2=pyhd3eb1b0_0 -y
conda install pygments=2.11.2=pyhd3eb1b0_0 -y
conda install python-dateutil=2.8.2=pyhd8ed1ab_0 -y
conda install python_abi=3.8=2_cp38 -y
conda install pytorch=1.12.0 -y		
conda install pytorch-mutex=1.0 -y
pip install pyzmq=19.0.2 -y				
conda install readline=8.1.2 -y
conda install setuptools=63.4.1 -y
conda install six=1.16.0=pyhd3eb1b0_1 -y
conda install sqlite=3.39.3 -y
conda install stack_data=0.2.0=pyhd3eb1b0_0 -y
conda install tk=8.6.12 -y
conda install tornado=6.1 -y
conda install tqdm=4.65.0 -y
conda install traitlets=5.1.1=pyhd3eb1b0_0 -y
conda install typing_extensions=4.3.0 -y
conda install wcwidth=0.2.5=pyhd3eb1b0_0 -y
conda install wheel=0.37.1=pyhd3eb1b0_0 -y
conda install xz=5.2.6 -y
conda install zeromq=4.3.4 -y
conda install zlib=1.2.12 -y
pip install numpy==1.23.4 -y
pip install pandas==1.5.0 -y
pip install pytz==2022.4 -y


#not needed - all (or equivalent) packages installed
#conda install entrypoints=0.4=pyhd8ed1ab_0
#conda install libsodium=1.0.18=h36c2ea0_1
#conda install pip=22.2.2=py38h06a4308_0
#conda install python=3.8.12=h12debd9_0


#missing both in conda and pip
#_libgcc_mutex=0.1=main
#_openmp_mutex=5.1=1_gnu
#cudatoolkit=10.2.89
#ld_impl_linux-64=2.38=h1181459_1
#libgcc-ng=11.2.0=h1234567_1
#libgomp=11.2.0=h1234567_1 			#available libgomp-amzn2-aarch64=7.3.1
#libstdcxx-ng=11.2.0=h1234567_1
#intel-openmp=2022.1.0=h9e868ea_3769
#mkl=2022.1.0=hc2b9512_224


set +x
echo "End"



