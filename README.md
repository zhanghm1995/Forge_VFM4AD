# Forge Vision Foundation Models for Autonoumous Driving: A Survey

**We greatly appreciate any contributions via PRs, issues, emails, or other methods.**

## Table of Content
- [Data Generation](#data-generation)
  - [GAN](#gan)

## Data Generation
### GAN
- **DriveGAN: Towards a Controllable High-Quality Neural Simulation**. 
    <details span>
    <summary><b>Abstract</b></summary>
    Realistic simulators are critical for training and verifying robotics systems. While most of the contemporary simulators are hand-crafted, a scaleable way to build simulators is to use machine learning to learn how the environment behaves in response to an action, directly from data. In this work, we aim to learn to simulate a dynamic environment directly in pixel-space, by watching unannotated sequences of frames and their associated action pairs. We introduce a novel high-quality neural simulator referred to as DriveGAN that achieves controllability by disentangling different components without supervision. In addition to steering controls, it also includes controls for sampling features of a scene, such as the weather as well as the location of non-player objects. Since DriveGAN is a fully differentiable simulator, it further allows for re-simulation of a given video sequence, offering an agent to drive through a recorded scene again, possibly taking different actions. We train DriveGAN on multiple datasets, including 160 hours of real-world driving data. We showcase that our approach greatly surpasses the performance of previous data-driven simulators, and allows for new features not explored before.

    <div align=center><img src="./assets/DriveGAN.png" width="70%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2104.15060-b31b1b.svg)](https://arxiv.org/abs/2104.15060) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](https://research.nvidia.com/labs/toronto-ai/DriveGAN/)


- **SurfelGAN: Synthesizing Realistic Sensor Data for Autonomous Driving**. 
    <details span>
    <summary><b>Abstract</b></summary>
    Autonomous driving system development is critically dependent on the ability to replay complex and diverse traffic scenarios in simulation. In such scenarios, the ability to accurately simulate the vehicle sensors such as cameras, lidar or radar is essential. However, current sensor simulators leverage gaming engines such as Unreal or Unity, requiring manual creation of environments, objects and material properties. Such approaches have limited scalability and fail to produce realistic approximations of camera, lidar, and radar data without significant additional work. In this paper, we present a simple yet effective approach to generate realistic scenario sensor data, based only on a limited amount of lidar and camera data collected by an autonomous vehicle. Our approach uses texture-mapped surfels to efficiently reconstruct the scene from an initial vehicle pass or set of passes, preserving rich information about object 3D geometry and appearance, as well as the scene conditions. We then leverage a SurfelGAN network to reconstruct realistic camera images for novel positions and orientations of the self-driving vehicle and moving objects in the scene. We demonstrate our approach on the Waymo Open Dataset and show that it can synthesize realistic camera data for simulated scenarios. We also create a novel dataset that contains cases in which two self-driving vehicles observe the same scene at the same time. We use this dataset to provide additional evaluation and demonstrate the usefulness of our SurfelGAN model.

    <div align=center><img src="./assets/SurfelGAN.png" width="70%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2005.03844-b31b1b.svg)](https://arxiv.org/abs/2005.03844)



👆 [Back to Top](#Table-of-Content)


### Diffusion
- **DrivingDiffusion: Layout-Guided multi-view driving scene video generation with latent diffusion model**.
    <details span>
    <summary><b>Abstract</b></summary>
    With the increasing popularity of autonomous driving based on the powerful and unified bird's-eye-view (BEV) representation, a demand for high-quality and large-scale multi-view video data with accurate annotation is urgently required. However, such large-scale multi-view data is hard to obtain due to expensive collection and annotation costs. To alleviate the problem, we propose a spatial-temporal consistent diffusion framework DrivingDiffusion, to generate realistic multi-view videos controlled by 3D layout. There are three challenges when synthesizing multi-view videos given a 3D layout: How to keep 1) cross-view consistency and 2) cross-frame consistency? 3) How to guarantee the quality of the generated instances? Our DrivingDiffusion solves the problem by cascading the multi-view single-frame image generation step, the single-view video generation step shared by multiple cameras, and post-processing that can handle long video generation. In the multi-view model, the consistency of multi-view images is ensured by information exchange between adjacent cameras. In the temporal model, we mainly query the information that needs attention in subsequent frame generation from the multi-view images of the first frame. We also introduce the local prompt to effectively improve the quality of generated instances. In post-processing, we further enhance the cross-view consistency of subsequent frames and extend the video length by employing temporal sliding window algorithm. Without any extra cost, our model can generate large-scale realistic multi-camera driving videos in complex urban scenes, fueling the downstream driving tasks. The code will be made publicly available.

    <div align=center><img src="./assets/DrivingDiffusion.png" width="70%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2310.07771-b31b1b.svg)](https://arxiv.org/abs/2310.07771) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](https://drivingdiffusion.github.io/) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/shalfun/DrivingDiffusion)


- **MagicDrive: Street View Generation with Diverse 3D Geometry Control**.
    <details span>
    <summary><b>Abstract</b></summary>
    Recent advancements in diffusion models have significantly enhanced the data synthesis with 2D control. Yet, precise 3D control in street view generation, crucial for 3D perception tasks, remains elusive. Specifically, utilizing Bird's-Eye View (BEV) as the primary condition often leads to challenges in geometry control (e.g., height), affecting the representation of object shapes, occlusion patterns, and road surface elevations, all of which are essential to perception data synthesis, especially for 3D object detection tasks. In this paper, we introduce MagicDrive, a novel street view generation framework offering diverse 3D geometry controls, including camera poses, road maps, and 3D bounding boxes, together with textual descriptions, achieved through tailored encoding strategies. Besides, our design incorporates a cross-view attention module, ensuring consistency across multiple camera views. With MagicDrive, we achieve high-fidelity street-view synthesis that captures nuanced 3D geometry and various scene descriptions, enhancing tasks like BEV segmentation and 3D object detection..

    <div align=center><img src="./assets/DrivingDiffusion.png" width="70%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2310.02601-b31b1b.svg)](https://arxiv.org/abs/2310.02601) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](https://flymin.github.io/magicdrive) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/cure-lab/MagicDrive)

👆 [Back to Top](#Table-of-Content)