execute the following commands (tested on WSL Ubuntu):

    sudo apt install make
    git clone --recurse-submodules https://github.com/TRI-ML/vidar.git
    cd vidar
    make docker-build 
    docker run --name running_vidar --mount type=bind,source=$HOME/data,target=/workspace/data --gpus all -it vidar_release

now inside the container, coning this repo:

    cd ..
    git clone https://github.com/kauevestena/monodepth_tests.git