# Install PyTorch & TensorFlow on AMD GPUs with ROCm (Ubuntu Linux Guide)

A beginner-friendly guide to installing PyTorch and TensorFlow on AMD GPUs with ROCm, tailored for users who are new to Linux. Includes step-by-step instructions to configure a development environment inside VS Code.

---

## Table of Contents:
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installing ROCm](#installing-rocm)
4. [Installing PyTorch and TensorFlow](#installing-pytorch-and-tensorflow)
5. [VS Code Setup for TensorFlow & PyTorch](#vs-code-setup-for-tensorflow--pytorch)
6. [Troubleshooting](#troubleshooting)  
7. [References](#references)  
8. [Contributing](#contributing)   
9. [Conclusion](#conclusion)  

---

## Introduction:
If you have an AMD graphics card and want to use machine learning libraries like PyTorch or TensorFlow, you may have noticed that the process isn’t as simple as running a quick pip install like it is with NVIDIA GPUs. To make it work, you’ll need a software stack called ROCm. Currently, ROCm is best supported on Linux, but the official documentation can often feel overwhelming, especially if you’re new to Linux and the terminal. Many existing guides assume prior experience, which makes the setup process confusing and frustrating for beginners. This guide is designed to walk you through the entire process step by step, from:
1. Installing ROCm
2. Setting up PyTorch and TensorFlow
3. Creating a fully working AI development environment inside Docker and VS Code

By the end, you’ll have your AMD GPU fully unlocked for deep learning development, with a clean and reliable setup that you can confidently use for your projects.

**YouTube Video Link:**

[![Watch the video](https://img.youtube.com/vi/pfnctTG5K2c/0.jpg)](https://www.youtube.com/watch?v=pfnctTG5K2c)

---

## System Requirements:
Check the official [AMD ROCm documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#system-requirements-linux)'s page.
> **Note:** If your GPU is not listed among the officially supported devices, it does not necessarily mean that ROCm cannot be installed on your system. It simply indicates that AMD has not provided official support for that model. In many cases, ROCm can still be installed and used on unsupported GPUs with additional configuration or tweaks.

---

## Installing ROCm:
1. Press CTRL + ALT + T simultaneously to open the terminal. Copy the following command and paste it on your terminal and then press Enter.
   
   - For 24.04:
   ```bash
   wget https://repo.radeon.com/amdgpu-install/6.4.3/ubuntu/noble/amdgpu-install_6.4.60403-1_all.deb
   sudo apt install ./amdgpu-install_6.4.60403-1_all.deb
   sudo apt update
   sudo apt install python3-setuptools python3-wheel
   sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
   sudo apt install rocm
   ```

   - For 22.04:
   ```bash
   wget https://repo.radeon.com/amdgpu-install/6.4.3/ubuntu/jammy/amdgpu-install_6.4.60403-1_all.deb
   sudo apt install ./amdgpu-install_6.4.60403-1_all.deb
   sudo apt update
   sudo apt install python3-setuptools python3-wheel
   sudo usermod -a -G render,video $LOGNAME # Add the current user to the render and video groups
   sudo apt install rocm
   ```
2. To install AMDGPU driver, run the following command.
   
   - For 24.04:
   ```bash
   wget https://repo.radeon.com/amdgpu-install/6.4.3/ubuntu/noble/amdgpu-install_6.4.60403-1_all.deb
   sudo apt install ./amdgpu-install_6.4.60403-1_all.deb
   sudo apt update
   sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
   sudo apt install amdgpu-dkms
   ```

   - For 22.04:
   ```bash
   wget https://repo.radeon.com/amdgpu-install/6.4.3/ubuntu/jammy/amdgpu-install_6.4.60403-1_all.deb
   sudo apt install ./amdgpu-install_6.4.60403-1_all.deb
   sudo apt update
   sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
   sudo apt install amdgpu-dkms
   ```
3. Now reboot your system to apply all the settings.
4. To configure ROCm shared objects, run the following command.

   ```bash
   sudo tee --append /etc/ld.so.conf.d/rocm.conf <<EOF
   /opt/rocm/lib
   /opt/rocm/lib64
   EOF
   sudo ldconfig
   ```
5. To configure ROCm PATH, use the update-alternatives utility by running this command.

   ```bash
   sudo update-alternatives --display rocm
   ```
6. To verify if ROCm was successfully installed, run the following command. If you see a bunch of information instead of an error, that means ROCm was successfully installed.

   ```bash
   rocminfo
   clinfo
   ```

---

## Installing PyTorch and TensorFlow:
1. First, install Docker if you don't have it already by running the following commands.

   ```bash
   sudo apt update
   sudo apt install docker.io
   sudo systemctl enable docker
   sudo systemctl start docker
   sudo usermod -aG docker $USER
   newgrp docker
   ```
2. Verify the docker installation using this command.
   ```bash
   docker ps
   ```
   If you see this output, **`CONTAINER ID IMAGE COMMAND CREATED STATUS PORTS NAMES`**. Then, your docker installation was successful.

3. Pull the latest PyTorch/TensorFlow Docker image.

   - For PyTorch:
   ```bash
   docker pull rocm/pytorch:latest
   ```

   - For TensorFlow:
   ```bash
   docker pull rocm/tensorflow:latest
   ```
4. Run it by using the following commands.

   - For PyTorch:
   ```bash
   docker run -it \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --device=/dev/kfd \
       --device=/dev/dri \
       --group-add video \
       --ipc=host \
       --shm-size 8G \
       rocm/pytorch:latest
   ```

   - For TensorFlow:
   ```bash
   docker run -it \
       --network=host \
       --device=/dev/kfd \
       --device=/dev/dri \
       --ipc=host \
       --shm-size 16G \
       --group-add video \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       rocm/tensorflow:latest
   ```
5. Verify the PyTorch/TensorFlow installation using the command below.

   - For PyTorch:
   ```bash
   python3 -c 'import torch' 2> /dev/null && echo 'Success' || echo 'Failure'
   ```

   - For TensorFlow:
   ```bash
   python -c 'import tensorflow' 2> /dev/null && echo ‘Success’ || echo ‘Failure’
   ```
   If you see 'Success', then you have successfully installed PyTorch/TensorFlow.

6. You can also run an example to see how your system is performing. Click [here](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/pytorch-install.html#running-a-basic-pytorch-example) for the code.

---

## VS Code Setup for TensorFlow & PyTorch:
1. In your project folder, create a new directory named **`.devcontainer`**.
2. Inside the **`.devcontainer`** directory, create a file called **`devcontainer.json`**.
3. Copy and paste the following code into **`devcontainer.json`**, then save the file.

   - For PyTorch:
   ```json
   {
     "name": "ROCm PyTorch Dev",
     "image": "rocm/pytorch:latest",
   
     "workspaceFolder": "/workspace",
     "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind",
   
     "runArgs": [
       "--cap-add=SYS_PTRACE",
       "--security-opt=seccomp=unconfined",
       "--device=/dev/kfd",
       "--device=/dev/dri",
       "--group-add=video",
       "--ipc=host",
       "--shm-size=8G"
     ],
   
     "customizations": {
       "vscode": {
         "extensions": [
           "ms-python.python",
           "ms-python.vscode-pylance"
         ]
       }
     },
   
     "settings": {
       "python.defaultInterpreterPath": "/usr/bin/python3"
     }
   }
   ```

   - For TensorFlow:
   ```json
   {
     "name": "ROCm TensorFlow Dev",
     "image": "rocm/tensorflow:latest",
   
     "workspaceFolder": "/workspace",
     "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind",
   
     "runArgs": [
       "--network=host",
       "--device=/dev/kfd",
       "--device=/dev/dri",
       "--ipc=host",
       "--shm-size=16G",
       "--group-add=video",
       "--cap-add=SYS_PTRACE",
       "--security-opt=seccomp=unconfined"
     ],
   
     "customizations": {
       "vscode": {
         "extensions": [
           "ms-python.python",
           "ms-python.vscode-pylance"
         ]
       }
     },
   
     "settings": {
       "python.defaultInterpreterPath": "/usr/bin/python3"
     }
   }
   ```
4. Open the Command Palette in VS Code by pressing **F1**, then search for and select **“Dev Containers: Open Folder in Container”**.
5. Choose the folder that contains the **`.devcontainer`** directory. VS Code will open the project inside the Dev Container, where PyTorch and TensorFlow are already pre-installed.
6. Ensure that you select the correct Python interpreter inside VS Code before running your code.
7. To confirm that everything is working correctly, try running the sample code snippets provided below.
   ```python
   import torch
   
   print("PyTorch version:", torch.__version__)
   print("GPU available:", torch.cuda.is_available())
   print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
   x = torch.rand(3, 3).cuda()
   print(x)
   ```
   ```python
   import tensorflow as tf
   print("TensorFlow version:", tf.__version__)
   mnist = tf.keras.datasets.mnist
   
   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   x_train, x_test = x_train / 255.0, x_test / 255.0
   model = tf.keras.models.Sequential([
     tf.keras.layers.Flatten(input_shape=(28, 28)),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dropout(0.2),
     tf.keras.layers.Dense(10)
   ])
   predictions = model(x_train[:1]).numpy()
   tf.nn.softmax(predictions).numpy()
   loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
   loss_fn(y_train[:1], predictions).numpy()
   model.compile(optimizer='adam',
                 loss=loss_fn,
                 metrics=['accuracy'])
   model.fit(x_train, y_train, epochs=5)
   model.evaluate(x_test,  y_test, verbose=2)
   ```

---

## Troubleshooting  
- **ROCm not detecting GPU**  
  Make sure your GPU is supported and that you’ve installed the correct ROCm version. You can confirm by running `rocminfo`.  

- **Docker permission issues**  
  Ensure your user is added to the `video` group and that Docker is installed correctly.  

- **VS Code Dev Container not starting**  
  Make sure the **Dev Containers** extension is installed and Docker is running in the background.  

- **Unsupported GPU warning**  
  Some GPUs work even if they’re not officially supported. Try using the latest ROCm version, but stability may vary.  

---

## References  
- [ROCm Documentation](https://rocm.docs.amd.com/)  
- [PyTorch ROCm](https://pytorch.org/get-started/locally/#rocm)  
- [TensorFlow ROCm](https://www.tensorflow.org/install/source_rocm)  
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)  

---

## Contributing  
Contributions are welcome!  
If you find bugs, have suggestions, or want to improve this guide, feel free to open an issue or submit a pull request.    

---

## Conclusion  
With this guide, you now have a fully functional machine learning environment on Ubuntu Linux using your AMD GPU with ROCm. Both PyTorch and TensorFlow should now be able to utilize GPU acceleration for your AI and deep learning projects.

If you run into issues, refer to the Troubleshooting section or check the official ROCm, PyTorch, and TensorFlow documentation.  
