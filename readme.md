# TVM Cartpole

This repository is a simple tutorial of TVM: Cartpole DQN at Pynq-z1 board with TVM compiler.  
For more tutorials, please refer this [TVM Tutorial site](https://tvm.apache.org/docs/tutorials/index.html). [VTA tutorial](https://tvm.apache.org/docs/vta/tutorials/index.html) is also important, but I won't cover VTA in this tutorial.  
Here, I used Pynq-z1 board for target hardware.  
I referred [Building a DQN in PyTorch: Balancing Cart Pole with Deep RL
](https://blog.gofynd.com/building-a-deep-q-network-in-pytorch-fa1086aa5435) for Cartpole DQN model. 
<br>
If you want to have in-depth information about TVM, I recommend you to read [ML2's TVM blog post](링크 넣기). Here, I wrote some troubleshootings that I experienced during this project.

<br>
<br>

## 0. Disclaimer
The final goal of this project is fully utilizing related TVM stacks like AutoTVM and VTA so that I can fully utilize Pynq-Z1 board's FPGA. However, due to time lack, I couldn't finish this code. Legacy of AutoTVM+VTA code is `vta_cartpole.py` file.  
Instead, you can refer `cpu_cartpole.py` file, which only utilizes CPU of Pynq-z1 board.

<br>
<br>

## 1. Installation
### Prerequisites
- Environment  
    - pynq-z1 board
    - MacBook Pro 2019
    - Ubuntu 18.04 with Virtualbox
    - Python 3.6
- Install python packges in `requirements.txt`(Recommend you to use virtual environments so that you don't get version conflict).  
    ```pip install -r requirements.txt```  
- TVM Installation
    - Since you can't install TVM with `pip` command, you have to install it from source.
    - In order to integrate the compiled module, we do not need to build entire TVM on the target device. You only need to build the TVM compiler stack on your desktop and use that to cross-compile modules that are deployed on the target device. We only need to use a light-weight runtime API that can be integrated into various platforms.
    - TVM(Host side)
        - Please follow [here](https://tvm.apache.org/docs/install/from_source.html) for TVM installation.
        - You have to set python properly so that you can run tvm.
        - Check whether TVM has successfully installed through importing it:  
            ```import tvm```
    - Runtime(Device side)
        - With cross compilation and RPC, you can compile a program on your local machine then run it on the remote device. It is useful when the remote device resource are limited. The runtime size is often less than 1MB, which makes it suitable for device with memory constraints.
        - Please follow [here](https://tvm.apache.org/docs/tutorials/get_started/cross_compilation_and_rpc.html) for runtime installation. 
        - After installation, run rpc-server at device side with:  
            ```python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090```
        - If you see this message, your device succesfully started RPC server:  
            ```INFO:root:RPCServer: bind to 0.0.0.0:9090```


<br>
<br>

## 2. File Description
```
 .
 ┣ 📂cartpole_model
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜cartpole-dqn.pth
 ┃ ┣ 📜dqn_agent.py
 ┃ ┣ 📜dqn_agent_demo.py
 ┃ ┣ 📜random_cartpole_agent.py
 ┃ ┗ 📜train_and_save.py
 ┣ 📜readme.md
 ┣ 📜requirements.txt
 ┣ 📜cpu_cartpole.py
 ┗ 📜vta_cartpole.py
```
- cartpole_model directory
    - Includes cartpole DQN model
    - Codes are derived from [Building a DQN in PyTorch: Balancing Cart Pole with Deep RL
](https://blog.gofynd.com/building-a-deep-q-network-in-pytorch-fa1086aa5435) 
    - `cartpole-dqn.pth`:
        - I already trained DQN model. You can train your own model with `train_and_save.py`, but if you don't want to, it's okay to use this file.
    - `dqn_agent.py`:
        - Defines DQN model
    - `dqn_agent_demo.py`:
        - Checks whether my DQN model works well
    - `random_cartpole_agent.py`:
        - Agent takes random action. Used to compare whether DQN model works well.
    - `train_and_save.py`:
        - Trains DQN model and save to `cartpole-dqn.pth`.
- `rpc_connect.py`
    - run cartpole at pynq board only with pynq-side-arm-cpu
- `vta_connect.py`  
    - incompleted
    - run cartpole at pynq board with fully utilizing FPGA resource.

<br>
<br>

## 3. Results
- First of all, you have to run rpc_server at pynq device:
```python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090```
- Run cartpole with:  
```python cpu_cartpole.py```
- Expected result:
```
Cannot find config for target=llvm -keys=cpu -mtriple=arm-linux-gnueabihf, workload=('dense_nopack.x86', ('TENSOR', (1, 64), 'float32'), ('TENSOR', (2, 64), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
Cannot find config for target=llvm -keys=cpu -mtriple=arm-linux-gnueabihf, workload=('dense_nopack.x86', ('TENSOR', (1, 4), 'float32'), ('TENSOR', (64, 4), 'float32'), None, 'float32'). A fallback configuration is used, which may bring great performance regression.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:04<00:00, 23.76it/s]
Test done
Elapsed time: 4.212365627288818 seconds
average reward per episode :9.32
```
- The warning messages, `Cannot find ~~ regression` occured because we don’t have tuned config for this model. Fallback means that the tunable parameters aren’t tuned, so the defaults will be used. Performance won’t be optimal. For more information, refer [this discussion](https://discuss.tvm.apache.org/t/what-does-this-warning-cannot-find-config-for-target-cuda-mean/798)