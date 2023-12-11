# Forge Vision Foundation Models for Autonoumous Driving: A Survey

**NOTE**: Here we select several featured papers for each part, and for each paper, we have contained the abstract and a figure from the original paper presenting the main framework or motivations to help us take a glance about these papers (You can expand the **Abstract** button to see them). More papers list and details can be found in our survey paper 🎉.

**We greatly appreciate any contributions via PRs, issues, emails, or other methods.**

## Table of Content
- [Related Survey Papers](#related-survey-papers)
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

## Related Survey Papers
- **On the Opportunities and Risks of Foundation Models**.
    <details span>
    <summary><b>Abstract</b></summary>
    AI is undergoing a paradigm shift with the rise of models (e.g., BERT, DALL-E, GPT-3) that are trained on broad data at scale and are adaptable to a wide range of downstream tasks. We call these models foundation models to underscore their critically central yet incomplete character. This report provides a thorough account of the opportunities and risks of foundation models, ranging from their capabilities (e.g., language, vision, robotics, reasoning, human interaction) and technical principles(e.g., model architectures, training procedures, data, systems, security, evaluation, theory) to their applications (e.g., law, healthcare, education) and societal impact (e.g., inequity, misuse, economic and environmental impact, legal and ethical considerations). Though foundation models are based on standard deep learning and transfer learning, their scale results in new emergent capabilities,and their effectiveness across so many tasks incentivizes homogenization. Homogenization provides powerful leverage but demands caution, as the defects of the foundation model are inherited by all the adapted models downstream. Despite the impending widespread deployment of foundation models, we currently lack a clear understanding of how they work, when they fail, and what they are even capable of due to their emergent properties. To tackle these questions, we believe much of the critical research on foundation models will require deep interdisciplinary collaboration commensurate with their fundamentally sociotechnical nature.

    <div align=center><img src="./assets/FM_Survey.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2108.07258-b31b1b.svg)](https://arxiv.org/abs/2108.07258) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](https://crfm.stanford.edu/report.html)

- **A Survey of Large Language Models for Autonomous Driving**.
    <details span>
    <summary><b>Abstract</b></summary>
    Autonomous driving technology, a catalyst for revolutionizing transportation and urban mobility, has the tend to transition from rule-based systems to data-driven strategies. Traditional module-based systems are constrained by cumulative errors among cascaded modules and inflexible pre-set rules. In contrast, end-to-end autonomous driving systems have the potential to avoid error accumulation due to their fully data-driven training process, although they often lack transparency due to their ``black box" nature, complicating the validation and traceability of decisions. Recently, large language models (LLMs) have demonstrated abilities including understanding context, logical reasoning, and generating answers. A natural thought is to utilize these abilities to empower autonomous driving. By combining LLM with foundation vision models, it could open the door to open-world understanding, reasoning, and few-shot learning, which current autonomous driving systems are lacking. In this paper, we systematically review a research line about \textit{Large Language Models for Autonomous Driving (LLM4AD)}. This study evaluates the current state of technological advancements, distinctly outlining the principal challenges and prospective directions for the field.

    <div align=center><img src="./assets/LLM4AD.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2304.10406-b31b1b.svg)](https://arxiv.org/abs/2304.10406) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/Thinklab-SJTU/Awesome-LLM4AD)

- **Applications of Large Scale Foundation Models for Autonomous Driving**.
    <details span>
    <summary><b>Abstract</b></summary>
    Since DARPA Grand Challenges (rural) in 2004/05 and Urban Challenges in 2007, autonomous driving has been the most active field of AI applications. Recently powered by large language models (LLMs), chat systems, such as chatGPT and PaLM, emerge and rapidly become a promising direction to achieve artificial general intelligence (AGI) in natural language processing (NLP). There comes a natural thinking that we could employ these abilities to reformulate autonomous driving. By combining LLM with foundation models, it is possible to utilize the human knowledge, commonsense and reasoning to rebuild autonomous driving systems from the current long-tailed AI dilemma. In this paper, we investigate the techniques of foundation models and LLMs applied for autonomous driving, categorized as simulation, world model, data annotation and planning or E2E solutions etc.
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2311.12144-b31b1b.svg)](https://arxiv.org/abs/2311.12144)

- **Vision Language Models in Autonomous Driving and Intelligent Transportation Systems**.
    <details span>
    <summary><b>Abstract</b></summary>
    TODOThe applications of Vision-Language Models (VLMs) in the fields of Autonomous Driving (AD) and Intelligent Transportation Systems (ITS) have attracted widespread attention due to their outstanding performance and the ability to leverage Large Language Models (LLMs). By integrating language data, the vehicles, and transportation systems are able to deeply understand real-world environments, improving driving safety and efficiency. In this work, we present a comprehensive survey of the advances in language models in this domain, encompassing current models and datasets. Additionally, we explore the potential applications and emerging research directions. Finally, we thoroughly discuss the challenges and research gap. The paper aims to provide researchers with the current work and future trends of VLMs in AD and ITS.

    <div align=center><img src="./assets/VLMs_in_AD.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2310.14414-b31b1b.svg)](https://arxiv.org/abs/2310.14414)

- **A Comprehensive Survey on Segment Anything Model for Vision and Beyond**.
    <details span>
    <summary><b>Abstract</b></summary>
    Artificial intelligence (AI) is evolving towards artificial general intelligence, which refers to the ability of an AI system to perform a wide range of tasks and exhibit a level of intelligence similar to that of a human being. This is in contrast to narrow or specialized AI, which is designed to perform specific tasks with a high degree of efficiency. Therefore, it is urgent to design a general class of models, which we term foundation models, trained on broad data that can be adapted to various downstream tasks. The recently proposed segment anything model (SAM) has made significant progress in breaking the boundaries of segmentation, greatly promoting the development of foundation models for computer vision. To fully comprehend SAM, we conduct a survey study. As the first to comprehensively review the progress of segmenting anything task for vision and beyond based on the foundation model of SAM, this work focuses on its applications to various tasks and data types by discussing its historical development, recent progress, and profound impact on broad applications. We first introduce the background and terminology for foundation models including SAM, as well as state-of-the-art methods contemporaneous with SAM that are significant for segmenting anything task. Then, we analyze and summarize the advantages and limitations of SAM across various image processing applications, including software scenes, real-world scenes, and complex scenes. Importantly, many insights are drawn to guide future research to develop more versatile foundation models and improve the architecture of SAM. We also summarize massive other amazing applications of SAM in vision and beyond.
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2305.08196-b31b1b.svg)](https://arxiv.org/abs/2305.08196) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/liliu-avril/Awesome-Segment-Anything)


👆 [Back to Top](#Table-of-Content)

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

- **Voxel-MAE - Masked Autoencoders for Self-Supervised Learning on Automotive Point Clouds**.
    <details span>
    <summary><b>Abstract</b></summary>
    Masked autoencoding has become a successful pretraining paradigm for Transformer models for text, images, and, recently, point clouds. Raw automotive datasets are suitable candidates for self-supervised pre-training as they generally are cheap to collect compared to annotations for tasks like 3D object detection (OD). However, the development of masked autoencoders for point clouds has focused solely on synthetic and indoor data. Consequently, existing methods have tailored their representations and models toward small and dense point clouds with homogeneous point densities. In this work, we study masked autoencoding for point clouds in an automotive setting, which are sparse and for which the point density can vary drastically among objects in the same scene. To this end, we propose Voxel-MAE, a simple masked autoencoding pre-training scheme designed for voxel representations. We pre-train the backbone of a Transformer-based 3D object detector to reconstruct masked voxels and to distinguish between empty and non-empty voxels. Our method improves the 3D OD performance by 1.75 mAP points and 1.05 NDS on the challenging nuScenes dataset. Further, we show that by pre-training with Voxel-MAE, we require only 40% of the annotated data to outperform a randomly initialized equivalent.

    <div align=center><img src="./assets/VoxelMAE.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2207.00531-b31b1b.svg)](https://arxiv.org/abs/2207.00531) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](https://georghess.se/projects/voxel-mae/) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/georghess/voxel-mae)

- **GD-MAE: Generative Decoder for MAE Pre-training on LiDAR Point Clouds**.
    <details span>
    <summary><b>Abstract</b></summary>
    Despite the tremendous progress of Masked Autoencoders (MAE) in developing vision tasks such as image and video, exploring MAE in large-scale 3D point clouds remains challenging due to the inherent irregularity. In contrast to previous 3D MAE frameworks, which either design a complex decoder to infer masked information from maintained regions or adopt sophisticated masking strategies, we instead propose a much simpler paradigm. The core idea is to apply a \textbf{G}enerative \textbf{D}ecoder for MAE (GD-MAE) to automatically merges the surrounding context to restore the masked geometric knowledge in a hierarchical fusion manner. In doing so, our approach is free from introducing the heuristic design of decoders and enjoys the flexibility of exploring various masking strategies. The corresponding part costs less than \textbf{12\%} latency compared with conventional methods, while achieving better performance. We demonstrate the efficacy of the proposed method on several large-scale benchmarks: Waymo, KITTI, and ONCE. Consistent improvement on downstream detection tasks illustrates strong robustness and generalization capability. Not only our method reveals state-of-the-art results, but remarkably, we achieve comparable accuracy even with \textbf{20\%} of the labeled data on the Waymo dataset.

    <div align=center><img src="./assets/GD-MAE.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2212.03010-b31b1b.svg)](https://arxiv.org/abs/2212.03010) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/Nightmare-n/GD-MAE)

- **UniM2AE: Multi-modal Masked Autoencoders with Unified 3D Representation for 3D Perception in Autonomous Driving**.
    <details span>
    <summary><b>Abstract</b></summary>
    Masked Autoencoders (MAE) play a pivotal role in learning potent representations, delivering outstanding results across various 3D perception tasks essential for autonomous driving. In real-world driving scenarios, it's commonplace to deploy multiple sensors for comprehensive environment perception. While integrating multi-modal features from these sensors can produce rich and powerful features, there is a noticeable gap in MAE methods addressing this integration. This research delves into multi-modal Masked Autoencoders tailored for a unified representation space in autonomous driving, aiming to pioneer a more efficient fusion of two distinct modalities. To intricately marry the semantics inherent in images with the geometric intricacies of LiDAR point clouds, the UniM$^2$AE is proposed. This model stands as a potent yet straightforward, multi-modal self-supervised pre-training framework, mainly consisting of two designs. First, it projects the features from both modalities into a cohesive 3D volume space, ingeniously expanded from the bird's eye view (BEV) to include the height dimension. The extension makes it possible to back-project the informative features, obtained by fusing features from both modalities, into their native modalities to reconstruct the multiple masked inputs. Second, the Multi-modal 3D Interactive Module (MMIM) is invoked to facilitate the efficient inter-modal interaction during the interaction process. Extensive experiments conducted on the nuScenes Dataset attest to the efficacy of UniM$^2$AE, indicating enhancements in 3D object detection and BEV map segmentation by 1.2\%(NDS) and 6.5\% (mIoU), respectively.

    <div align=center><img src="./assets/UniM2AE.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](https://arxiv.org/abs/2304.10406) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](TODO) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](TODO)


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

- **ALSO: Automotive Lidar Self-supervision by Occupancy estimation**.
    <details span>
    <summary><b>Abstract</b></summary>
    We propose a new self-supervised method for pre-training the backbone of deep perception models operating on point clouds. The core idea is to train the model on a pretext task which is the reconstruction of the surface on which the 3D points are sampled, and to use the underlying latent vectors as input to the perception head. The intuition is that if the network is able to reconstruct the scene surface, given only sparse input points, then it probably also captures some fragments of semantic information, that can be used to boost an actual perception task. This principle has a very simple formulation, which makes it both easy to implement and widely applicable to a large range of 3D sensors and deep networks performing semantic segmentation or object detection. In fact, it supports a single-stream pipeline, as opposed to most contrastive learning approaches, allowing training on limited resources. We conducted extensive experiments on various autonomous driving datasets, involving very different kinds of lidars, for both semantic segmentation and object detection. The results show the effectiveness of our method to learn useful representations without any annotation, compared to existing approaches.

    <div align=center><img src="./assets/ALSO.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2212.05867-b31b1b.svg)](https://arxiv.org/abs/2212.05867) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/valeoai/ALSO)

👆 [Back to Top](#Table-of-Content)

### Contrastive
- **BEVContrast: Self-Supervision in BEV Space for Automotive Lidar Point Clouds**.
    <details span>
    <summary><b>Abstract</b></summary>
    We present a surprisingly simple and efficient method for self-supervision of 3D backbone on automotive Lidar point clouds. We design a contrastive loss between features of Lidar scans captured in the same scene. Several such approaches have been proposed in the literature from PointConstrast [40 ], which uses a contrast at the level of points, to the state-of-the-art TARL [30 ], which uses a contrast at the level of segments, roughly corresponding to objects. While the former enjoys a great simplicity of implementation, it is surpassed by the latter, which however requires a costly pre-processing. In BEVContrast, we define our contrast at the level of 2D cells in the Bird's Eye View plane. Resulting cell-level representations offer a good trade-off between the point-level representations exploited in PointContrast and segment-level representations exploited in TARL: we retain the simplicity of PointContrast (cell representations are cheap to compute) while surpassing the performance of TARL in downstream semantic segmentation.

    <div align=center><img src="./assets/BEVContrast.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2310.17281-b31b1b.svg)](https://arxiv.org/abs/2310.17281) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/valeoai/BEVContrast)

- **SegContrast: 3D Point Cloud Feature Representation Learning Through Self-Supervised Segment Discrimination**.
    <details span>
    <summary><b>Abstract</b></summary>
    Semantic scene interpretation is essential for autonomous systems to operate in complex scenarios. While deep learning-based methods excel at this task, they rely on vast amounts of labeled data that is tedious to generate and might not cover all relevant classes sufficiently. Self-supervised representation learning has the prospect of reducing the amount of required labeled data by learning descriptive representations from unlabeled data. In this letter, we address the problem of representation learning for 3D point cloud data in the context of autonomous driving. We propose a new contrastive learning approach that aims at learning the structural context of the scene. Our approach extracts class-agnostic segments over the point cloud and applies the contrastive loss over these segments to discriminate between similar and dissimilar structures. We apply our method on data recorded with a 3D LiDAR. We show that our method achieves competitive performance and can learn a more descriptive feature representation than other state-of-the-art self-supervised contrastive point cloud methods.

    <div align=center><img src="./assets/SegContrast.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/IEEE-ICRA-b31b1b.svg)](https://www.ipb.uni-bonn.de/pdfs/nunes2022ral-icra.pdf) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/PRBonn/segcontrast)

- **Temporal Consistent 3D LiDAR Representation Learning for Semantic Perception in Autonomous Driving**.
    <details span>
    <summary><b>Abstract</b></summary>
    Semantic perception is a core building block in autonomous driving, since it provides information about the drivable space and location of other traffic participants. For learning-based perception, often a large amount of diverse training data is necessary to achieve high performance. Data labeling is usually a bottleneck for developing such methods, especially for dense prediction tasks, e.g., semantic segmentation or panoptic segmentation. For 3D LiDAR data, the annotation process demands even more effort than for images. Especially in autonomous driving, point clouds are sparse, and objects appearance depends on its distance from the sensor, making it harder to acquire large amounts of labeled training data. This paper aims at taking an alternative path proposing a self-supervised representation learning method for 3D LiDAR data. Our approach exploits the vehicle motion to match objects across time viewed in different scans. We then train a model to maximize the point-wise feature similarities from points of the associated object in different scans, which enables to learn a consistent representation across time. The experimental results show that our approach performs better than previous state-of-the-art self-supervised representation learning methods when fine-tuning to different downstream tasks. We furthermore show that with only 10% of labeled data, a network pre-trained with our approach can achieve better performance than the same network trained from scratch with all labels for semantic segmentation on SemanticKITTI.

    <div align=center><img src="./assets/TARL.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/CVF-CVPR-6196CA.svg)](https://openaccess.thecvf.com/content/CVPR2023/html/Nunes_Temporal_Consistent_3D_LiDAR_Representation_Learning_for_Semantic_Perception_in_CVPR_2023_paper.html)  [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/PRBonn/TARL)


- **PointContrast: Unsupervised Pre-training for 3D Point Cloud Understanding**.
    <details span>
    <summary><b>Abstract</b></summary>
    Arguably one of the top success stories of deep learning is transfer learning. The finding that pre-training a network on a rich source set (e.g., ImageNet) can help boost performance once fine-tuned on a usually much smaller target set, has been instrumental to many applications in language and vision. Yet, very little is known about its usefulness in 3D point cloud understanding. We see this as an opportunity considering the effort required for annotating data in 3D. In this work, we aim at facilitating research on 3D representation learning. Different from previous works, we focus on high-level scene understanding tasks. To this end, we select a suite of diverse datasets and tasks to measure the effect of unsupervised pre-training on a large source set of 3D scenes. Our findings are extremely encouraging: using a unified triplet of architecture, source dataset, and contrastive loss for pre-training, we achieve improvement over recent best results in segmentation and detection across 6 different benchmarks for indoor and outdoor, real and synthetic datasets – demonstrating that the learned representation can generalize across domains. Furthermore, the improvement was similar to supervised pre-training, suggesting that future efforts should favor scaling data collection over more detailed annotation. We hope these findings will encourage more research on unsupervised pretext task design for 3D deep learning.

    <div align=center><img src="./assets/PointContrast.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2007.10985-b31b1b.svg)](https://arxiv.org/abs/2007.10985) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/facebookresearch/PointContrast)

👆 [Back to Top](#Table-of-Content)

### Rendering
- **Ponder: Point Cloud Pre-training via Neural Rendering**.
    <details span>
    <summary><b>Abstract</b></summary>
    We propose a novel approach to self-supervised learning of point cloud representations by differentiable neural rendering. Motivated by the fact that informative point cloud features should be able to encode rich geometry and appearance cues and render realistic images, we train a point-cloud encoder within a devised point-based neural renderer by comparing the rendered images with real images on massive RGB-D data. The learned point-cloud encoder can be easily integrated into various downstream tasks, including not only high-level tasks like 3D detection and segmentation, but low-level tasks like 3D reconstruction and image synthesis. Extensive experiments on various tasks demonstrate the superiority of our approach compared to existing pre-training methods.

    <div align=center><img src="./assets/Ponder.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2301.00157-b31b1b.svg)](https://arxiv.org/abs/2301.00157)
    
- **PRED: Pre-training via Semantic Rendering on LiDAR Point Clouds**.
    <details span>
    <summary><b>Abstract</b></summary>
    Pre-training is crucial in 3D-related fields such as autonomous driving where point cloud annotation is costly and challenging. Many recent studies on point cloud pre-training, however, have overlooked the issue of incompleteness, where only a fraction of the points are captured by LiDAR, leading to ambiguity during the training phase. On the other hand, images offer more comprehensive information and richer semantics that can bolster point cloud encoders in addressing the incompleteness issue inherent in point clouds. Yet, incorporating images into point cloud pre-training presents its own challenges due to occlusions, potentially causing misalignments between points and pixels. In this work, we propose PRED, a novel image-assisted pre-training framework for outdoor point clouds in an occlusion-aware manner. The main ingredient of our framework is a Birds-Eye-View (BEV) feature map conditioned semantic rendering, leveraging the semantics of images for supervision through neural rendering. We further enhance our model's performance by incorporating point-wise masking with a high mask ratio (95%). Extensive experiments demonstrate PRED's superiority over prior point cloud pre-training methods, providing significant improvements on various large-scale datasets for 3D perception tasks.

    <div align=center><img src="./assets/PRED.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2311.04501-b31b1b.svg)](https://arxiv.org/abs/2311.04501) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/PRED4pc/PRED)

- **UniPAD: A Universal Pre-training Paradigm for Autonomous Driving**.
    <details span>
    <summary><b>Abstract</b></summary>
    In the context of autonomous driving, the significance of effective feature learning is widely acknowledged. While conventional 3D self-supervised pre-training methods have shown widespread success, most methods follow the ideas originally designed for 2D images. In this paper, we present UniPAD, a novel self-supervised learning paradigm applying 3D volumetric differentiable rendering. UniPAD implicitly encodes 3D space, facilitating the reconstruction of continuous 3D shape structures and the intricate appearance characteristics of their 2D projections. The flexibility of our method enables seamless integration into both 2D and 3D frameworks, enabling a more holistic comprehension of the scenes. We manifest the feasibility and effectiveness of UniPAD by conducting extensive experiments on various downstream 3D tasks. Our method significantly improves lidar-, camera-, and lidar-camera-based baseline by 9.1, 7.7, and 6.9 NDS, respectively. Notably, our pre-training pipeline achieves 73.2 NDS for 3D object detection and 79.4 mIoU for 3D semantic segmentation on the nuScenes validation set, achieving state-of-the-art results in comparison with previous methods.

    <div align=center><img src="./assets/UniPAD.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2310.08370-b31b1b.svg)](https://arxiv.org/abs/2310.08370) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/Nightmare-n/UniPAD)


- **RenderOcc: Vision-Centric 3D Occupancy Prediction with 2D Rendering Supervision**.
    <details span>
    <summary><b>Abstract</b></summary>
    3D occupancy prediction holds significant promise in the fields of robot perception and autonomous driving, which quantifies 3D scenes into grid cells with semantic labels. Recent works mainly utilize complete occupancy labels in 3D voxel space for supervision. However, the expensive annotation process and sometimes ambiguous labels have severely constrained the usability and scalability of 3D occupancy models. To address this, we present RenderOcc, a novel paradigm for training 3D occupancy models only using 2D labels. Specifically, we extract a NeRF-style 3D volume representation from multi-view images, and employ volume rendering techniques to establish 2D renderings, thus enabling direct 3D supervision from 2D semantics and depth labels. Additionally, we introduce an Auxiliary Ray method to tackle the issue of sparse viewpoints in autonomous driving scenarios, which leverages sequential frames to construct comprehensive 2D rendering for each object. To our best knowledge, RenderOcc is the first attempt to train multi-view 3D occupancy models only using 2D labels, reducing the dependence on costly 3D occupancy annotations. Extensive experiments demonstrate that RenderOcc achieves comparable performance to models fully supervised with 3D labels, underscoring the significance of this approach in real-world applications.

    <div align=center><img src="./assets/RenderOcc.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2309.09502-b31b1b.svg)](https://arxiv.org/abs/2309.09502) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/pmj110119/RenderOcc)

- **SelfOcc: Self-Supervised Vision-Based 3D Occupancy Prediction**.
    <details span>
    <summary><b>Abstract</b></summary>
    3D occupancy prediction is an important task for the robustness of vision-centric autonomous driving, which aims to predict whether each point is occupied in the surrounding 3D space. Existing methods usually require 3D occupancy labels to produce meaningful results. However, it is very laborious to annotate the occupancy status of each voxel. In this paper, we propose SelfOcc to explore a self-supervised way to learn 3D occupancy using only video sequences. We first transform the images into the 3D space (e.g., bird's eye view) to obtain 3D representation of the scene. We directly impose constraints on the 3D representations by treating them as signed distance fields. We can then render 2D images of previous and future frames as self-supervision signals to learn the 3D representations. We propose an MVS-embedded strategy to directly optimize the SDF-induced weights with multiple depth proposals. Our SelfOcc outperforms the previous best method SceneRF by 58.7% using a single frame as input on SemanticKITTI and is the first self-supervised work that produces reasonable 3D occupancy for surround cameras on nuScenes. SelfOcc produces high-quality depth and achieves state-of-the-art results on novel depth synthesis, monocular depth estimation, and surround-view depth estimation on the SemanticKITTI, KITTI-2015, and nuScenes, respectively.

    <div align=center><img src="./assets/SelfOcc.png" width="100%" /></div>
    </details>

    [![arXiv](https://img.shields.io/badge/arXiv-2311.12754-b31b1b.svg)](https://arxiv.org/abs/2311.12754) [![WEB Page](https://img.shields.io/badge/Project-Page-159957.svg)](https://huang-yh.github.io/SelfOcc/) [![WEB Page](https://img.shields.io/badge/Github-Page-159957.svg)](https://github.com/huang-yh/SelfOcc)

### World Model


👆 [Back to Top](#Table-of-Content)