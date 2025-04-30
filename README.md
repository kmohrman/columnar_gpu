# columnar_gpu

Example `srun` command to get a node with GPU (at UF):
```
srun --qos=avery --account=avery --partition=gpu --gpus=1 --mem=16000 --constraint=a100 --pty bash -i
```
Example `srun` command to get a node with CPU (at UF):
```
srun -t 600 --qos=avery --account=avery --cpus-per-task=4 --mem-per-cpu=4G --pty bash -i
```

Set up the environment:
```
conda env create -f environment.yml
conda activate coffeagpu_env
pip install hepconvert
conda uninstall coffea
```
Then navigate to your local coffea dir from Lindsey (get it via `git clone -b jitters https://github.com/scikit-hep/coffea.git`) and pip install it.

Now you are ready to run.
