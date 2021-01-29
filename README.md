# VTA Cartpole
The main backbone code is [Auto-tuning a convolutional network on VTA](https://tvm.apache.org/docs/vta/tutorials/autotvm/tune_relay_vta.html#sphx-glr-vta-tutorials-autotvm-tune-relay-vta-py), which deals with conv2d operation optimization.  
This branch is incompleted project. I wrote some stuffs that should be modified below.  

## Environment Settings
Make sure your connection between pynq board and host machine.  
At host side, run this command to open tracker:  
`python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190`  
The expected output is:  
`INFO:RPCTracker:bind to 0.0.0.0:9190`  
Then, run this command at pynq side to open server:  
`python -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=pynq`  
Modify [HOST_IP] part to your one. My one is 192.168.0.13.  
When your device registeration is done, confirm it by querying the rpc_tracker:  
`python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190`
Your expected result is:  
```
Tracker address 0.0.0.0:9190

Server List
----------------------------
server-address  key
----------------------------
192.168.0.36:53142      server:pynq
----------------------------

Queue Status
----------------------------
key    total  free  pending
----------------------------
pynq   1      1     0      
----------------------------
```
You can add many devices if you have.


## Stuffs that should be modified:
**1. Graphpacking**  
When you see `tvm/vta/python/vta/top/graphpack.py/get_subgraph` function, you can see this comment:
```
    """We assume stop_name only appears once for simplicity.
    This constraint will be lifted in the future.
    bitpack_start and bitpack_end are both inclusive.
    ""
```
However, in my code, stop_name(add in my case) appears twice. Below is my model derived from relay:  
```
def @main(%cartpole-input: Tensor[(1, 4), float32], %v0.weight: Tensor[(64, 4), float32], %v0.bias: Tensor[(64), float32], %v2.weight: Tensor[(2, 64), float32], %v2.bias: Tensor[(2), float32]) {
  %0 = transpose(%v0.weight, axes=[1, 0]);
  %1 = transpose(%0, axes=[1, 0]);
  %2 = nn.dense(%cartpole-input, %1, units=64);
  %3 = add(%2, %v0.bias);
  %4 = tanh(%3);
  %5 = transpose(%v2.weight, axes=[1, 0]);
  %6 = transpose(%5, axes=[1, 0]);
  %7 = nn.dense(%4, %6, units=2);
  %8 = add(%7, %v2.bias);
  nn.softmax(%8, axis=0)
}
```
**2. register_vta_tuning_tasks()**  
I wrote some code with referring conv2d code, but I have to debug it after finishing graph_packing part.  

**3. Extracting task part**  
We have task extraction part.  
`tasks = list(filter(lambda t: len(t.args[0][1]) > 4, tasks))`  
I have to modify this part to adopt dense operation.  
To do so, I have to check extracted tasks.