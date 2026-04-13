# SIMPLE Teleop

VR teleoperation in SIMPLE. Built upon modifications to the [GR00T decoupled  WBC codebase](https://nvlabs.github.io/GR00T-WholeBodyControl/references/decoupled_wbc.html), it allows for  whole-body control and data collection via a PICO VR headset.

## 🛠 Installation

Please ensure that the main SIMPLE environment is successfully installed before proceeding. For detailed environment setup, please refer to the **SIMPLE installation**.

## 🥽 VR Teleop Setup
The following setup process refers to the [GR00T-WholeBodyControl VR Teleop Setup](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html).

### 1. Required Hardware

* **PICO 4 / PICO 4 Ultra** headset
* **2x PICO controllers**
* **2x PICO motion trackers**

<img src="../../../assets/img/Required_Hardware.jpg" alt="Required Hardware Setup" height="300" width="300">

##  2. Install XRoboToolkit

XRoboToolkit bridges the connection between your workstation and the VR headset. It consists of a PC service and a PICO application.

### 2.1. PC Service (Host Machine)
1. Navigate to the [XR-Robotics GitHub repository](https://github.com/XR-Robotics).
2. Follow the **"Install XRoboToolkit-PC-Service"** instructions provided there to install the service on your workstation.

<img src="../../../assets/img/Xtoolrobokit_pc.png" alt="XRoboToolkit PC Service Setup" width="600">

### 2. PICO App (VR Headset)
Put on your PICO headset and complete the following steps inside the VR environment:

1. Complete the initial quick setup and ensure the headset is connected to **Wi-Fi**.
2. *(Optional)* Ensure **Developer Mode** is enabled (`Settings` → `Developer`). *Note: It is generally okay if you cannot find this setting.*
3. Open the built-in **Browser** application.
4. Type **"xrobotoolkit"** in the search bar and navigate to the XR-Robotics GitHub page.
5. Scroll down to the releases section and use the PICO trigger to download the APK file: **`XRoboToolkit-PICO-1.1.1.apk`**.

<img src="../../../assets/img/xrobotoolkit_pico.png" alt="XRoboToolkit PICO App Download" width="600">

6. Open the **Manage Downloads** section (located at the top right corner of the browser page) and click on the downloaded `.apk` file.
7. Select **Install**. Once the installation is complete, the application will appear in the **Unknown** section of your App Library.


<img src="../../../assets/img/xrobotoolkit.jpg" alt="XRoboToolkit PICO App Setup" width="600">

----
# 🕹️ Teleop Tutorial
🎉 **Ready to Go!** Once you have successfully set up SIMPLE environment and installed the XRoboToolkit on both your host machine and PICO headset, your hardware and software are fully equipped. **You can now start performing teleoperation to collect data within SIMPLE!** Let's dive into the operational steps below. 👇

### 1. Launch the Teleoperation Script

To start the teleoperation server and load a specific simulation environment, open your terminal and run the following command from the root of your `SIMPLE` directory:

```bash
python src/simple/cli/teleop_decoupled_wbc.py simple/G1WholebodyXMoveBendPickTeleop-v0 
--target=graspnet1b:0 --sim-mode=mujoco --record --no-headless
```


**Command Arguments Explained:**

* **`env_id` (The Environment):** This is the first argument (e.g., `simple/G1WholebodyXMoveBendPickTeleop-v0`). It defines the task scene. You can replace it with any of our currently supported environments:
    * `simple/G1WholebodyXMoveBendPickTeleop-v0` 
    * `simple/G1WholebodyPickAndPlaceAndHugContainerTeleop-v0`
    * `simple/G1WholebodyLocomotionPickBetweenTablesSonic-v0`
    * `simple/G1WholebodyHandoverSonic-v0` 

* **`--target` (The Object):** Specifies the interaction target using the format `<dataset_name>:<object_id>`.
    * **Modify ID:** Change the integer to load different objects (e.g., `--target=graspnet1b:12`).
    * **Modify Dataset:** Switch the source from `graspnet1b` to `objaverse` (e.g., `--target=objaverse:5`).


###  2. PICO Motion Tracker Setup

<img src="../../../assets/img/pico_motion_tracker.png" alt="Pico Motion Tracker Setup" width="600">

##### **2.1 Attachment & Visibility**
 Strap one PICO motion tracker to your **left ankle** and one to your **right ankle**.Scrunch down any baggy clothing to ensure the trackers are fully visible to the headset cameras. Ensure the side with the **light indicator** faces **upward**.

##### **2.2 Accessing Tracker Menu**
1.  In the PICO main menu, click the **Wi-Fi icon**.
2.  A device overview will appear. Look for the **small circular motion tracker logo** above the headset icon. 
    * *Note: If the logo is missing, manually open the **"Motion Tracker"** system app.*
3.  Ensure the headset and two controllers are populated, then select the **Motion Tracker** (small circle).

##### **2.3 Pairing Procedure**
1.  **Clear Old Devices:** Click the **"i"** icon next to each currently listed tracker and select **Unpair**.
2.  **Enter Pairing Mode:** Once cleared, click the **"Pair"** button in the top right corner.
3.  **Sync Trackers:** Press and hold the button on top of each tracker for **6 seconds**. The indicator lights will flash **red and blue** when they enter pairing mode.

##### **2.4 Motion Tracker Calibration**
Wear the PICO headset properly over your eyes and press the blue **"Calibrate"** button. Follow these two sequences:

* **Sequence 1:** Stand perfectly stiff with your arms and handheld controllers straight down by your sides.
* **Sequence 2:** Look down at the foot trackers until the headset cameras successfully recognize and lock onto them.

Once the calibration sequence is complete, you should see a virtual avatar appear . 
<img src="../../../assets/img/pico_calibrate.jpg" alt="Pico Motion Tracker Calibration" width="600">

### 3. PICO XRoboToolkit Setup


**Pre-requisite:** Ensure both your **Host PC** (running the Teleoperation Script) and the **PICO headset** are connected to the **same Local Area Network**.

1.  **Open XRoboToolkit:** Launch the app from the **Unknown Sources** section of your PICO library.
2.  **Auto-Detection:** In most cases, the app will automatically detect your PC's IP address on the network.
<img src="../../../assets/img/xrobotoolkit_ip.jpg" alt="XRoboToolkit IP Address Detection" width="600">

3.  **Establish Connection:** * Locate your PC's IP address on the screen as shown in the image below.
    * Click the **Connect** button.
4.  **Verify Status:** Before proceeding to the next step, ensure the connection **Status** displays as **`Working`**.

After the connection status shows **Working**, you need to configure the specific data streams to synchronize your movements with the robot in SIMPLE.

1.  **Tracking Configuration:**
    * Under the **Tracking** section, ensure both **Head** and **Controller** are selected.
    * Set the **PICO Motion Tracker** mode to **Full-body**.
2.  **Data & Control:**
    * Enable the **Send** switch.
3.  **Remote Vision:**
    * Locate the **Remote Vision** section.
    * Set the **State** (or stream source) to **Zedmini**.

Once configured correctly, your interface should match the following setup:
<img src="../../../assets/img/xrobotoolkit_config.jpg" alt="XRoboToolkit Configuration" width="600">

After configuring the tracking and data control, the final step is to sync the visual feedback so you can see through the robot's "eyes."

1. **Activate Listen:** In the **Remote Vision** section, click the **Listen** button. 
2. **Visual Feedback:** You should now see the real-time visual stream from the robot's head cameras as rendered in the **SIMPLE** simulation environment.
3. **Toggle View Mode:** By default, the stream may show a **dual-camera (stereo) view** (left and right eye feeds). You can press the **B button** on your **Right Controller** to toggle the view mode:

#### **Stereo (Dual) Camera View**
In this mode, you see the output of both the left and right sensors from the robot's Zedmini camera:
<img src="../../../assets/img/teleop_two_camera.jpg" alt="PICO Stereo Camera View" width="600">

#### **Single Camera View (Toggle with B Button)**
Switching to single view provides a unified field of vision:
<img src="../../../assets/img/teleop_one_view.jpg" alt="PICO Single Camera View" width="600">

---

**🚀 Success!** You are now fully connected. Now you can start controlling the robot.


## 🎮 Controller Mapping

Once the environment is running and the visual stream is active, use the following mapping to operate the robot. 

> **⚠️ Important:** At the start of each episode, please **wait 4-6 seconds** for the robot to stabilize its stance before engaging teleoperation.

### ⌨️ Controls Reference Table

| Action | Controller Input | Description |
| :--- | :--- | :--- |
| **Start Teleoperation** | `Left Menu` + `Right index Trigger` | **Left Menu:** Button with three lines.<br>**Right index Trigger:** Right index finger. |
| **Locomotion (X/Y)** | `Left Joystick` | Move the robot base (Forward/Backward/Sidestep). |
| **Rotation (Yaw)** | `Right Joystick` | Rotate the robot base (Turn Left/Right). |
| **Left Hand** | `Left index Trigger` | **Left index finger:** Controls opening/closing of the left hand. |
| **Right Hand** | `Right index Trigger` | **Right index finger:** Controls opening/closing of the right hand. |
| **Squat / Crouch** | `X Button` | **Left Controller:** Robot performs a squatting motion. |
| **Stand Up** | `Y Button` | **Right Controller:** Robot returns to a standing posture. |
| **Manual Reset** | `Left Middle Trigger` + `Right Middle Trigger` | **Middle Fingers:** Press both side buttons  to force-reset episode. |
| **Toggle View** | `B Button` | **Right Controller:** Toggle between Stereo and Single camera views. |

---

### 🚀 Operation Workflow

1.  **Stability Phase:** After the simulation loads, let the robot stand still for **4-6 seconds** to reach a stable state.
2.  **Engage Control:** Hold `Left Menu` and `Right indexTrigger` simultaneously. Your physical posture will now sync to the robot.
3.  **Execution & Saving:**
    * **Auto-Save:** If the system judges the task as **Successful**, it will automatically save the trajectory and refresh the episode.
    * **Manual Reset:** If you wish to restart or if the task fails, press **both Side middle trigger** (middle fingers) simultaneously to reset the episode manually.
