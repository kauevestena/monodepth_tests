## execute the following commands (tested on WSL Ubuntu):

    sudo apt install make
    git clone --recurse-submodules https://github.com/TRI-ML/vidar.git
    cd vidar
    make docker-build 
    docker run --name running_vidar --mount type=bind,source=$HOME/data,target=/workspace/data --gpus all -it vidar_release

## now inside the container, clone this very repo:

    cd ..
    git clone https://github.com/kauevestena/monodepth_tests.git

### to test performance and download the models into cache, run the tests:

    python tri_vidar/test_packnet.py
    python tri_vidar/test_zerodepth.py