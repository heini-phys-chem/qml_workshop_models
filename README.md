# Scripts for QMLightning for the Workshop

## install:
 - install torch (`pip install torch`)
 - install cuda12-2 using (choose your system settings):
```
https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local
```
and put following stuff at the end of your .bashrc:
```
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.2
```
 - make sure you have the correct nvidia drivers (type in tetrminal: `nvidia-smi` or `nvcc`)
 - clone the git repo (https://github.com/nickjbrowning/qml-lightning) and install it using:
 ```
 pip install . --no-build-isolation -v
 ```


## usage:

### Train
```
python predict.py -ntrain <path to train npz> -ntest <path to test/validation npz> -npcas 128 -nbatch_train 64 -nbatch_test 128
```

npcas should be kept at 128 to recover the entire representation. If not enough memory is available on the GPU, try to reduce nbatch_train and nbatch_test.
The hyperparameter scan can be activated with `-hyperparam_opt 1`.


### MD

```
python run_MD.py
```

Some key things: The filename of the model which is loaded is usually hardcoded!, Make sure you have a `xyz` directory to store the coordinates of every nth step.
The calculator python file for QMLightning has to be stored in ASE calculator directory.


### making GIFs using Jmol
Store all xyz coordinates into one xyz file, open it with jmol. In jmol go to `file/console`. In the console type:
```
script trajectorymoevie.jmol
```
This will write all xyz frames into .jpg files. Then you can use the ubuntu native `convert` to make a GIF:
```
convert -delay 10 -loop 0 *.jpg output.gif
```
