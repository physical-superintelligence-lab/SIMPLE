# Teleoperation using PICO


## Bring up 

1. In terminal 1: start the simulator
```
python src/simple/cli/teleop.py simple/G1WholebodyBendPick-v1
```

2. In terminal 2: start the SONIC controller

> Install SONIC first following the [official documentation](https://nvlabs.github.io/GR00T-WholeBodyControl/)

```
cd ~/workspace/projects/gr00t-wbc
cd gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh sim --input-type zmq_manager
```

3. In terminal 3: start the PICO manager server

```
cd third_party
python gear_sonic/scripts/pico_manager_thread_server.py \
    --manager --vis_vr3pt --vis_smpl
```

## Operation

>  Checkout the official document for more operation modes: [PICO VR Whole-body Teleop](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html)


After `bringing up`:

+ First, press `A+B+X+Y` to engage `SONIC` controller.  The humanoid will begain to kick and swing.

+ Second, press `right joystick click` to lower the humanoid onto the ground.

+ Lastly, press `A+X` to enter `POSE` mode.

> Make sure to stand in `calibration pose` before entering `POSE` mode.
  
+ Press `left grip + right grip` to reset the environment anytime.


## Troubleshootings

[TODO]
 