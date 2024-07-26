# RLBench to UniRobo

This repository transfers RLBench tasks and demostrations into UniRobo, a framework based on NVIDIA IsaacSim. It may also inspire future migration of other tasks and demonstrations developed based on RLBench (PyRep (CoppeliaSim)).

## Installation

Follow the [guidance](https://github.com/stepjam/RLBench?tab=readme-ov-file#install) in RLBench repository to install CoppeliaSim v4.1.0, PyRep, and RLBench.

In addition, PyRep does not support gravity configuration. Therefore, to change the gravity, you should add the following code at the end of `pyrep/backend/simAddOnScript_PyRep.lua` in your PyRep repo. Installing PyRep in editable mode with `-e` may help. The `tools/replay_object_states.py` script requires this function to enable zero gravity.
```lua
setGravity=function(inInts,inFloats,inStrings,inBuffer)
    local gravity = {inFloats[1], inFloats[2], inFloats[3]}
    sim.setArrayParameter(sim.arrayparam_gravity, gravity)
    return {},{},{},''
end
```

## Usage

To get prepared, run `setup_pyrep`. You may need to change the pathes in the scripts.

Note: PyRep (CoppeliaSim) will launch Qt even if in headless mode. To ensure it runs successfully, run `export DISPLAY=:1` before you run the following command if you are using remote SSH.

- To collect demostrations in order to replay in UniRobo, run
    ```
    python tools/collect_demo.py --episode_num=2 --headless --task=close_box
    ```

- To collect demostrations in order to replay in CoppliaSim without robot arm, run
    ```
    python tools/collect_demo.py --episode_num=2 --headless --task=close_box --record_object_states
    ```

- To replay demostrations in CoppeliaSim **without** robot arm, and capture visual observations, run
    ```
    python tools/replay_object_states.py --headless --task close_box
    ```
    The recorded data will be in `outputs` folder.

- To replay demostrations in CoppeliaSim **with** robot arm, and capture visual observations, run
    ```
    python tools/replay_object_states.py --headless --task close_box --with_robot
    ```
    The recorded data will be in `outputs` folder.
    TODO: Fix the bug for the shadow of the robot arm.

- To see all available tasks, run the following command:
    ```bash
    cat data/cfg/rlbench_objects.yaml | yq -r "keys[]"
    ```

- To check if all tasks have right configuration, run the following command:
    ```bash
    cat data/cfg/rlbench_objects.yaml | yq "to_entries | map({key: .key, subkeys: .value | keys})"
    ```
    Each task should have 'objects' and 'joints' properties.

### Pipeline Usage

Change tasks in `generate_config_batch.sh` and run the following commands to generate and start tmux sessions.

```bash
bash generate_config_batch.sh > config_batch.yml
tmuxinator start -p config_batch.yml
```
