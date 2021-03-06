from cartpole_model import DQN_Agent
import gym
import numpy as np
import torch
import time
from tqdm import tqdm
import os

import tvm
from tvm import te
from tvm import rpc
from tvm import relay
from tvm.contrib import utils
from tvm.contrib import graph_runtime


# Environment Setup
# Check target env with "gcc -v" command at device
# Turn on rpc server at pynq side with command:
# `python -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090`
# You can see this message that shows your rpc server is working well
# INFO:root:RPCServer: bind to 0.0.0.0:9090

target = 'llvm -mtriple=arm-linux-gnueabihf' # my device is pynq-z1 board
host = '192.168.0.36' # xilinx ip address. Modify to your proper address
port = 9091
assert tvm.runtime.enabled("rpc")
remote = rpc.connect(host, port)


# Load a pretrained PyTorch model
world = gym.make('CartPole-v0')
input_dim = world.observation_space.shape[0]
output_dim = world.action_space.n
exp_replay_size = 256
model = DQN_Agent(seed=1423, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=5,
                  exp_replay_size=exp_replay_size)


# Load model
model_path = os.path.join(os.getcwd(), 'cartpole_model', 'cartpole-dqn.pth')
model.load_pretrained_model(model_path)
model = model.q_net.eval()


# Grab the TorchScripted model via tracing
rand_input = torch.from_numpy(world.reset()).float().unsqueeze(0)
scripted_model = torch.jit.trace(model, rand_input).eval()


# Import PyTorch graph to Relay.
input_name = "cartpole-input"
shape_list = [(input_name, rand_input.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)


# Build Relay graph
# Compile the graph to llvm target with given input specification.
ctx = tvm.cpu()
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)


# Save the compiled module library to the disk(at local temporary directory,
# which will be uploaded to the remote machine later.
tmp = tvm.contrib.utils.tempdir()
lib_fname = tmp.relpath("lib.tar")
lib.export_library(lib_fname)


# upload the library to remote device and load it
remote.upload(lib_fname)
rlib = remote.load_module("lib.tar")


# create remote runtime module
ctx = remote.cpu(0)
module = graph_runtime.GraphModule(rlib["default"](ctx))


# run cartpole
start_time = time.time()
reward_arr = []
for i in tqdm(range(100)):
    obs, done, total_reward = world.reset(), False, 0
    while not done:
        module.set_input(input_name, tvm.nd.array(np.expand_dims(obs.astype("float32"), axis=0)))
        module.run()
        out = module.get_output(0).asnumpy()
        A = np.argmax(out)
        obs, reward, done, info = world.step(A.item())
        total_reward += reward
    reward_arr.append(total_reward)

elapsed_time = time.time() - start_time


# Print results
print("Test done")
print("Elapsed time: {} seconds".format(elapsed_time))
print("average reward per episode :{}".format(sum(reward_arr) / len(reward_arr)))