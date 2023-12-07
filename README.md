# Forge Vision Foundation Models for Autonoumous Driving: A Survey

**NOTE**: Here we select several featured papers for each part, and for each paper, we have contained the abstract and a figure from the original paper presenting the main framework or motivations to help us take a glance about these papers (You can expand the **Abstract** button to see them). More papers list and details can be found in our survey paper 🎉.

**We greatly appreciate any contributions via PRs, issues, emails, or other methods.**

## Table of Content
- [Data Generation](#data-generation)
  - [GAN](#gan)
  - [Diffusion](#diffusion)
  - [NeRF](#nerf)
- [Training Paradigm](#training-paradigm)
  - [Reconstruction](#reconstruction)
  - [Contrastive](#contrastive)
  - [Rendering](#rendering)
  - [Distillation](#distillation)
  - [World Model](#world-model)

## Data Generation
### GAN
- **DriveGAN: Towards a Controllable High-Quality Neural Simulation**. 
    <details span>
    <summary><b>Abstract</b></summary>
    Realistic simulators are critical for training and verifying robotics systems. While most of the contemporary simulators are hand-crafted, a scaleable way to build simulators is to use machine learning to learn how the environment behaves in response to an action, directly from data. In this work, we aim to learn to simulate a dynamic environment directly in pixel-space, by watching unannotated sequences of frames and their associated action pairs. We introduce a novel high-quality neural simulator referred to as DriveGAN that achieves controllability by disentangling different components without supervision. In addition to steering controls, it also includes controls for sampling features of a scene, such as the weather as well as the location of non-player objects. Since DriveGAN is a fully differentiable simulator, it further allows for re-simulation of a given video sequence, offering an agent to drive through a recorded scene again, possibly taking different actions. We train DriveGAN on multiple datasets, including 160 hours of real-world driving data. We showcase that our approach greatly surpasses the performance of previous data-driven simulators, and allows for new features not explored before.

    <div align=center><img src="./assets/DriveGAN.png" width="80%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2104.15060-b31b1b.svg)](https://arxiv.org/abs/2104.15060) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](https://research.nvidia.com/labs/toronto-ai/DriveGAN/)


- **SurfelGAN: Synthesizing Realistic Sensor Data for Autonomous Driving**. 
    <details span>
    <summary><b>Abstract</b></summary>
    Autonomous driving system development is critically dependent on the ability to replay complex and diverse traffic scenarios in simulation. In such scenarios, the ability to accurately simulate the vehicle sensors such as cameras, lidar or radar is essential. However, current sensor simulators leverage gaming engines such as Unreal or Unity, requiring manual creation of environments, objects and material properties. Such approaches have limited scalability and fail to produce realistic approximations of camera, lidar, and radar data without significant additional work. In this paper, we present a simple yet effective approach to generate realistic scenario sensor data, based only on a limited amount of lidar and camera data collected by an autonomous vehicle. Our approach uses texture-mapped surfels to efficiently reconstruct the scene from an initial vehicle pass or set of passes, preserving rich information about object 3D geometry and appearance, as well as the scene conditions. We then leverage a SurfelGAN network to reconstruct realistic camera images for novel positions and orientations of the self-driving vehicle and moving objects in the scene. We demonstrate our approach on the Waymo Open Dataset and show that it can synthesize realistic camera data for simulated scenarios. We also create a novel dataset that contains cases in which two self-driving vehicles observe the same scene at the same time. We use this dataset to provide additional evaluation and demonstrate the usefulness of our SurfelGAN model.

    <div align=center><img src="./assets/SurfelGAN.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2005.03844-b31b1b.svg)](https://arxiv.org/abs/2005.03844)



👆 [Back to Top](#Table-of-Content)


### Diffusion
- **DrivingDiffusion: Layout-Guided multi-view driving scene video generation with latent diffusion model**.
    <details span>
    <summary><b>Abstract</b></summary>
    With the increasing popularity of autonomous driving based on the powerful and unified bird's-eye-view (BEV) representation, a demand for high-quality and large-scale multi-view video data with accurate annotation is urgently required. However, such large-scale multi-view data is hard to obtain due to expensive collection and annotation costs. To alleviate the problem, we propose a spatial-temporal consistent diffusion framework DrivingDiffusion, to generate realistic multi-view videos controlled by 3D layout. There are three challenges when synthesizing multi-view videos given a 3D layout: How to keep 1) cross-view consistency and 2) cross-frame consistency? 3) How to guarantee the quality of the generated instances? Our DrivingDiffusion solves the problem by cascading the multi-view single-frame image generation step, the single-view video generation step shared by multiple cameras, and post-processing that can handle long video generation. In the multi-view model, the consistency of multi-view images is ensured by information exchange between adjacent cameras. In the temporal model, we mainly query the information that needs attention in subsequent frame generation from the multi-view images of the first frame. We also introduce the local prompt to effectively improve the quality of generated instances. In post-processing, we further enhance the cross-view consistency of subsequent frames and extend the video length by employing temporal sliding window algorithm. Without any extra cost, our model can generate large-scale realistic multi-camera driving videos in complex urban scenes, fueling the downstream driving tasks. The code will be made publicly available.

    <div align=center><img src="./assets/DrivingDiffusion.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2310.07771-b31b1b.svg)](https://arxiv.org/abs/2310.07771) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](https://drivingdiffusion.github.io/) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/shalfun/DrivingDiffusion)


- **MagicDrive: Street View Generation with Diverse 3D Geometry Control**.
    <details span>
    <summary><b>Abstract</b></summary>
    Recent advancements in diffusion models have significantly enhanced the data synthesis with 2D control. Yet, precise 3D control in street view generation, crucial for 3D perception tasks, remains elusive. Specifically, utilizing Bird's-Eye View (BEV) as the primary condition often leads to challenges in geometry control (e.g., height), affecting the representation of object shapes, occlusion patterns, and road surface elevations, all of which are essential to perception data synthesis, especially for 3D object detection tasks. In this paper, we introduce MagicDrive, a novel street view generation framework offering diverse 3D geometry controls, including camera poses, road maps, and 3D bounding boxes, together with textual descriptions, achieved through tailored encoding strategies. Besides, our design incorporates a cross-view attention module, ensuring consistency across multiple camera views. With MagicDrive, we achieve high-fidelity street-view synthesis that captures nuanced 3D geometry and various scene descriptions, enhancing tasks like BEV segmentation and 3D object detection.

    <div align=center><img src="./assets/MagicDrive.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2310.02601-b31b1b.svg)](https://arxiv.org/abs/2310.02601) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](https://flymin.github.io/magicdrive) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/cure-lab/MagicDrive)


- **DatasetDM: Synthesizing Data with Perception Annotations Using Diffusion Models**.
    <details span>
    <summary><b>Abstract</b></summary>
    Current deep networks are very data-hungry and benefit from training on largescale datasets, which are often time-consuming to collect and annotate. By contrast, synthetic data can be generated infinitely using generative models such as DALL-E and diffusion models, with minimal effort and cost. In this paper, we present DatasetDM, a generic dataset generation model that can produce diverse synthetic images and the corresponding high-quality perception annotations (e.g., segmentation masks, and depth). Our method builds upon the pre-trained diffusion model and extends text-guided image synthesis to perception data generation. We show that the rich latent code of the diffusion model can be effectively decoded as accurate perception annotations using a decoder module. Training the decoder only needs less than 1% (around 100 images) manually labeled images, enabling the generation of an infinitely large annotated dataset. Then these synthetic data can be used for training various perception models for downstream tasks. To showcase the power of the proposed approach, we generate datasets with rich dense pixel-wise labels for a wide range of downstream tasks, including semantic segmentation, instance segmentation, and depth estimation. Notably, it achieves 1) state-of-the-art results on semantic segmentation and instance segmentation; 2) significantly more robust on domain generalization than using the real data alone; and state-of-the-art results in zero-shot segmentation setting; and 3) flexibility for efficient application and novel task composition (e.g., image editing).

    <div align=center><img src="./assets/DatasetDM.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2308.06160-b31b1b.svg)](https://arxiv.org/abs/2308.06160) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](https://weijiawu.github.io/DatasetDM_page/) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/showlab/DatasetDM)

👆 [Back to Top](#Table-of-Content)


### NeRF
- **UniSim: Synthesizing Data with Perception Annotations Using Diffusion Models**.
    <details span>
    <summary><b>Abstract</b></summary>
    Rigorously testing autonomy systems is essential for making safe self-driving vehicles (SDV) a reality. It requires one to generate safety critical scenarios beyond what can be collected safely in the world, as many scenarios happen rarely on public roads. To accurately evaluate performance, we need to test the SDV on these scenarios in closed-loop, where the SDV and other actors interact with each other at each timestep. Previously recorded driving logs provide a rich resource to build these new scenarios from, but for closed loop evaluation, we need to modify the sensor data based on the new scene configuration and the SDV's decisions, as actors might be added or removed and the trajectories of existing actors and the SDV will differ from the original log. In this paper, we present UniSim, a neural sensor simulator that takes a single recorded log captured by a sensor-equipped vehicle and converts it into a realistic closed-loop multi-sensor simulation. UniSim builds neural feature grids to reconstruct both the static background and dynamic actors in the scene, and composites them together to simulate LiDAR and camera data at new viewpoints, with actors added or removed and at new placements. To better handle extrapolated views, we incorporate learnable priors for dynamic objects, and leverage a convolutional network to complete unseen regions. Our experiments show UniSim can simulate realistic sensor data with small domain gap on downstream tasks. With UniSim, we demonstrate closed-loop evaluation of an autonomy system on safety-critical scenarios as if it were in the real world.

    <div align=center><img src="./assets/UniSim.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2308.01898-b31b1b.svg)](https://arxiv.org/abs/2308.01898) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](https://waabi.ai/unisim/)

- **MARS: An Instance-aware, Modular and Realistic Simulator for Autonomous Driving**.
    <details span>
    <summary><b>Abstract</b></summary>
    Nowadays, autonomous cars can drive smoothly in ordinary cases, and it is widely recognized that realistic sensor simulation will play a critical role in solving remaining corner cases by simulating them. To this end, we propose an autonomous driving simulator based upon neural radiance fields (NeRFs). Compared with existing works, ours has three notable features: (1) Instance-aware. Our simulator models the foreground instances and background environments separately with independent networks so that the static (e.g., size and appearance) and dynamic (e.g., trajectory) properties of instances can be controlled separately. (2) Modular. Our simulator allows flexible switching between different modern NeRF-related backbones, sampling strategies, input modalities, etc. We expect this modular design to boost academic progress and industrial deployment of NeRF-based autonomous driving simulation. (3) Realistic. Our simulator set new state-of-the-art photo-realism results given the best module selection. Our simulator will be open-sourced while most of our counterparts are not.

    <div align=center><img src="./assets/MARS.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2307.15058-b31b1b.svg)](https://arxiv.org/abs/2307.15058) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](https://open-air-sun.github.io/mars/) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/OPEN-AIR-SUN/mars)

- **NeRF-LiDAR: Generating Realistic LiDAR Point Clouds with Neural Radiance Fields**.
    <details span>
    <summary><b>Abstract</b></summary>
    Labeling LiDAR point clouds for training autonomous driving is extremely expensive and difficult. LiDAR simulation aims at generating realistic LiDAR data with labels for training and verifying self-driving algorithms more efficiently. Recently, Neural Radiance Fields (NeRF) have been proposed for novel view synthesis using implicit reconstruction of 3D scenes. Inspired by this, we present NeRF-LIDAR, a novel LiDAR simulation method that leverages real-world information to generate realistic LIDAR point clouds. Different from existing LiDAR simulators, we use real images and point cloud data collected by self-driving cars to learn the 3D scene representation, point cloud generation and label rendering. We verify the effectiveness of our NeRF-LiDAR by training different 3D segmentation models on the generated LiDAR point clouds. It reveals that the trained models are able to achieve similar accuracy when compared with the same model trained on the real LiDAR data. Besides, the generated data is capable of boosting the accuracy through pre-training which helps reduce the requirements of the real labeled data.

    <div align=center><img src="./assets/NeRF-LiDAR.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2304.14811-b31b1b.svg)](https://arxiv.org/abs/2304.14811)

- **LiDAR-NeRF: Novel LiDAR View Synthesis via Neural Radiance Fields**.
    <details span>
    <summary><b>Abstract</b></summary>
    We introduce a new task, novel view synthesis for LiDAR sensors. While traditional model-based LiDAR simulators with style-transfer neural networks can be applied to render novel views, they fall short of producing accurate and realistic LiDAR patterns because the renderers rely on explicit 3D reconstruction and exploit game engines, that ignore important attributes of LiDAR points. We address this challenge by formulating, to the best of our knowledge, the first differentiable end-to-end LiDAR rendering framework, LiDAR-NeRF, leveraging a neural radiance field (NeRF) to facilitate the joint learning of geometry and the attributes of 3D points. However, simply employing NeRF cannot achieve satisfactory results, as it only focuses on learning individual pixels while ignoring local information, especially at low texture areas, resulting in poor geometry. To this end, we have taken steps to address this issue by introducing a structural regularization method to preserve local structural details. To evaluate the effectiveness of our approach, we establish an object-centric multi-view LiDAR dataset, dubbed NeRF-MVL. It contains observations of objects from 9 categories seen from 360-degree viewpoints captured with multiple LiDAR sensors. Our extensive experiments on the scene-level KITTI-360 dataset, and on our object-level NeRF-MVL show that our LiDAR-NeRF surpasses the model-based algorithms significantly.

    <div align=center><img src="./assets/LiDAR-NeRF.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2304.10406-b31b1b.svg)](https://arxiv.org/abs/2304.10406) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](https://tangtaogo.github.io/lidar-nerf-website/) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/tangtaogo/lidar-nerf)

👆 [Back to Top](#Table-of-Content)

## Training Paradigm
### Reconstruction

- **UniScene: Multi-Camera Unified Pre-training via 3D Scene Reconstruction**.
    <details span>
    <summary><b>Abstract</b></summary>
    Multi-camera 3D perception has emerged as a prominent research field in autonomous driving, offering a viable and cost-effective alternative to LiDAR-based solutions. The existing multi-camera algorithms primarily rely on monocular 2D pre-training. However, the monocular 2D pre-training overlooks the spatial and temporal correlations among the multi-camera system. To address this limitation, we propose the first multi-camera unified pre-training framework, called UniScene, which involves initially reconstructing the 3D scene as the foundational stage and subsequently fine-tuning the model on downstream tasks. Specifically, we employ Occupancy as the general representation for the 3D scene, enabling the model to grasp geometric priors of the surrounding world through pre-training. A significant benefit of UniScene is its capability to utilize a considerable volume of unlabeled image-LiDAR pairs for pre-training purposes. The proposed multi-camera unified pre-training framework demonstrates promising results in key tasks such as multi-camera 3D object detection and surrounding semantic scene completion. When compared to monocular pre-training methods on the nuScenes dataset, UniScene shows a significant improvement of about 2.0% in mAP and 2.0% in NDS for multi-camera 3D object detection, as well as a 3% increase in mIoU for surrounding semantic scene completion. By adopting our unified pre-training method, a 25% reduction in 3D training annotation costs can be achieved, offering significant practical value for the implementation of real-world autonomous driving.

    <div align=center><img src="./assets/UniScene.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2305.18829-b31b1b.svg)](https://arxiv.org/abs/2305.18829) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/chaytonmin/UniScene)

- **Occupancy-MAE: Self-supervised Pre-training Large-scale LiDAR Point Clouds with Masked Occupancy Autoencoders**.
    <details span>
    <summary><b>Abstract</b></summary>
    Current perception models in autonomous driving heavily rely on large-scale labelled 3D data, which is both costly and time-consuming to annotate. This work proposes a solution to reduce the dependence on labelled 3D training data by leveraging pre-training on large-scale unlabeled outdoor LiDAR point clouds using masked autoencoders (MAE). While existing masked point autoencoding methods mainly focus on small-scale indoor point clouds or pillar-based large-scale outdoor LiDAR data, our approach introduces a new self-supervised masked occupancy pre-training method called Occupancy-MAE, specifically designed for voxel-based large-scale outdoor LiDAR point clouds. Occupancy-MAE takes advantage of the gradually sparse voxel occupancy structure of outdoor LiDAR point clouds and incorporates a range-aware random masking strategy and a pretext task of occupancy prediction. By randomly masking voxels based on their distance to the LiDAR and predicting the masked occupancy structure of the entire 3D surrounding scene, Occupancy-MAE encourages the extraction of high-level semantic information to reconstruct the masked voxel using only a small number of visible voxels. Extensive experiments demonstrate the effectiveness of Occupancy-MAE across several downstream tasks. For 3D object detection, Occupancy-MAE reduces the labelled data required for car detection on the KITTI dataset by half and improves small object detection by approximately 2% in AP on the Waymo dataset. For 3D semantic segmentation, Occupancy-MAE outperforms training from scratch by around 2% in mIoU. For multi-object tracking, Occupancy-MAE enhances training from scratch by approximately 1% in terms of AMOTA and AMOTP.

    <div align=center><img src="./assets/OccupancyMAE.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2206.09900-b31b1b.svg)](https://arxiv.org/abs/2206.09900) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/chaytonmin/Occupancy-MAE)

👆 [Back to Top](#Table-of-Content)