# Install PyTorch & TensorFlow on AMD GPUs with ROCm (Ubuntu Linux Guide)

A beginner-friendly guide to installing PyTorch and TensorFlow on AMD GPUs with ROCm, tailored for users who are new to Linux. Includes step-by-step instructions to configure a development environment inside VS Code.

---

## Table of Contents:
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installing ROCm](#installing-rocm)
4. [Installing PyTorch/TensorFlow](#installing-pytorch-tensorflow)
5. [Docker Setup for TensorFlow & PyTorch](#docker-setup-for-tensorflow--pytorch)
6. [Native Installation](#native-installation)
7. [Testing the Installation](#testing-the-installation)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)

---

## Introduction:
If you have an AMD graphics card and want to use machine learning libraries like PyTorch or TensorFlow, you may have noticed that the process isn’t as simple as running a quick pip install like it is with NVIDIA GPUs. To make it work, you’ll need a software stack called ROCm. Currently, ROCm is best supported on Linux, but the official documentation can often feel overwhelming, especially if you’re new to Linux and the terminal. Many existing guides assume prior experience, which makes the setup process confusing and frustrating for beginners. This guide is designed to walk you through the entire process step by step, from:
1. Installing ROCm
2. Setting up PyTorch and TensorFlow
3. Creating a fully working AI development environment inside Docker and VS Code

By the end, you’ll have your AMD GPU fully unlocked for deep learning development, with a clean and reliable setup that you can confidently use for your projects.

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

## Installing PyTorch/TensorFlow:
