from cartpole_model import DQN_Agent
import gym
import numpy as np
import torch
import time
from tqdm import tqdm
import os

import tvm
from tvm import relay, rpc, te, autotvm, topi
from tvm.contrib import utils
from tvm.contrib import graph_runtime

import vta
from vta.top import graph_pack


#################################################################
# Start RPC Tracker
# -----------------
# TVM uses an RPC session to communicate with Pynq boards.
# During tuning, the tuner will send the generated code to the board and
# measure the speed of code on the board.
#
# To start an RPC tracker, run this command on the host machine. The tracker is
# required during the whole tuning process, so we need to open a new terminal for
# this command:
# `python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190`
#
# The expected output is:
# `INFO:RPCTracker:bind to 0.0.0.0:9190`

#################################################################
# Register devices to RPC Tracker
# -----------------------------------
# Register the device to the tracker with:
# `python -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=pynq`
# (replace :code:`[HOST_IP]` with the IP address of your host machine)
#
# After registering devices, we can confirm it by querying the rpc_tracker:
# `python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190`
# You can register multiple devices to the tracker to accelerate tuning.


### Environment Setup
# Parameters for model
world = gym.make('CartPole-v0')
input_dim = world.observation_space.shape[0]
output_dim = world.action_space.n
exp_replay_size = 256

# Designate device
env = vta.get_env() # Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
device = "vta"  # arm_cpu to run inference on CPU, and vta to run inference on the FPGA
target = env.target if device == "vta" else env.target_vta_cpu
target_host = env.target_host

# Make sure that TVM was compiled with RPC=1
assert tvm.runtime.enabled("rpc")

# Options for tracker
tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))

# Tuning option
log_file = "{}-{}.log".format(device, "cartpole")
tuning_option = {
    "log_filename": log_file,
    "tuner": "random",
    "n_trial": 1000,
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.RPCRunner(
            env.TARGET,
            host=tracker_host,
            port=tracker_port,
            number=5,
            timeout=60,
            # check_correctness=True, # TODO: re-enable when check_correctness works again.
        ),
    ),
}



def compile_cartpole_model(env, start_pack, stop_pack, start_name_idx, stop_name_idx):

    # Load cartpole model
    model = DQN_Agent(seed=1423, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=5,
                  exp_replay_size=exp_replay_size)
    model_path = os.path.join(os.getcwd(), 'cartpole_model', 'cartpole-dqn.pth')
    model.load_pretrained_model(model_path)
    model = model.q_net.eval()

    # Grab the TorchScripted model via tracing 
    rand_input = torch.from_numpy(world.reset()).float().unsqueeze(0)
    scripted_model = torch.jit.trace(model, rand_input).eval() 

    dtype_dict = {"data": "float32"}
    shape_dict = {"data": (1,4)}

    # Import the graph to Relay
    # Convert PyTorch graph to Relay graph. The input name can be arbitrary.
    input_name = "cartpole-input"
    shape_list = [(input_name, rand_input.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    print(mod)

    # Update shape and type dictionary
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Perform quantization in Relay
    # Note: We set opt_level to 3 in order to fold batch norm
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)

        # Perform graph packing and constant folding for VTA target
        assert env.BLOCK_IN == env.BLOCK_OUT
        
        # !!! Here is where I get suffered now !!!
        # Graph packing for VTA
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=start_pack,
            stop_name=stop_pack,
            start_name_idx=start_name_idx,
            stop_name_idx=stop_name_idx
        )
    return relay_prog, params




def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Register VTA-specific tuning tasks
# Didn't debugged yet. Have to complete graphpack part first.
def register_vta_tuning_tasks():
    from tvm.autotvm.task import TaskExtractEnv

    @tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
    def my_clip(x, a_min, a_max):
        """Unlike topi's current clip, put min and max into two stages."""
        const_min = tvm.tir.const(a_min, x.dtype)
        const_max = tvm.tir.const(a_max, x.dtype)
        x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
        x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
        return x

    # init autotvm env to register VTA operator
    TaskExtractEnv()

    @autotvm.template("dense_packed.vta")
    def _topi_nn_dense(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A, W, b = args[:3]

        with tvm.target.vta():
            res = vta.top.dense_packed(*args, **kwargs)
            res = topi.right_shift(res, 8)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")
    
        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.schedule_dense_packed([res])
        
        return s, [A, W, b, res]



def tune_and_evaluate(tuning_opt):

    # Register VTA tuning tasks
    register_vta_tuning_tasks()

    # Perform task extraction on Relay program
    print("Extract tasks...")
    relay_prog, params = compile_cartpole_model(env=env, start_pack="nn.dense", stop_pack="add", start_name_idx=None, stop_name_idx=8)
    mod = tvm.IRModule.from_expr(relay_prog)
    tasks = autotvm.task.extract_from_program(
        mod,
        params=params,
        ops=(relay.op.get("nn.dense"),),
        target=target,
        target_host=env.target_host,
    )

    # Have to adopt this part for dense operation
    # filter out non-packed conv2d task
    tasks = list(filter(lambda t: len(t.args[0][1]) > 4, tasks))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # evaluate with tuning history
    # Get remote from fleet node
    remote = autotvm.measure.request_remote(
        env.TARGET, tracker_host, tracker_port, timeout=10000
    )
    # Reconfigure the JIT runtime and FPGA.
    vta.reconfig_runtime(remote)
    vta.program_fpga(remote, bitstream=None)

    # compile kernels with history best records
    with autotvm.tophub.context(target, extra_files=[log_file]):
        # Compile network
        print("Compile...")
        with vta.build_config(opt_level=3):
            lib = relay.build(relay_prog, target=target, params=params, target_host=env.target_host)

        # Export library
        # Send the inference library over to the remote RPC server
        temp = utils.tempdir()
        lib.export_library(temp.relpath("graphlib.tar"))
        remote.upload(temp.relpath("graphlib.tar"))
        lib = remote.load_module("graphlib.tar")

        # Graph runtime
        module = graph_runtime.GraphModule(lib["default"](ctx))


    # run cartpole
    start_time = time.time()
    reward_arr = []
    for i in tqdm(range(100)):
        obs, done, rew = world.reset(), False, 0
        while not done:
            module.set_input(input_name, tvm.nd.array(np.expand_dims(obs.astype("float32"), axis=0)))
            module.run()
            out = m.get_output(0).asnumpy()
            A = np.argmax(out)
            obs, reward, done, info = world.step(A.item())
            total_reward += reward
        reward_arr.append(total_reward)

    elapsed_time = time.time() - start_time

    print("Test done")
    print("Elapsed time: {} seconds".format(elapsed_time))
    print("average reward per episode :{}".format(sum(reward_arr) / len(reward_arr)))



tune_and_evaluate(tuning_option)