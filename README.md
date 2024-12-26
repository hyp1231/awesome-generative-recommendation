# Awesome Generative Recommendation

[![PR Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/hyp1231/awesome-generative-recommendation/pulls)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

- [Papers](#papers)
  - [Overview & Surveys](#overview--surveys)
  - [LLM-based Generative Recommendation](#llm-based-generative-recommendation)
    - [Zero-shot Recommendation with LLMs](#zero-shot-recommendation-with-llms)
    - [Aligning LLMs with User Behaivors](#aligning-llms-with-user-behaviors)
    - [LLM-powered Agents in Recommendation](#llm-powered-agents-in-recommendation)
    - [LLM-based Conversational Recommender Systems](#llm-based-conversational-recommender-systems)
  - [Semantic ID-based Generative Recommendation](#semantic-id-based-generative-recommendation)
    - [Generative Retrieval and Semantic IDs](#generative-retrieval-and-semantic-ids)
    - [Semantic ID-based Recommender Architecture](#semantic-id-based-recommender-architecture)
    - [Semantic ID Construction](#semantic-id-construction)
  - [Diffusion Model-based Generative Recommendation](#diffusion-model-based-generative-recommendation)
    - [Diffusion-based Recommendation Architecture](#diffusion-based-recommendation-architecture)
    - [ID Embedding Generation with Diffusion](#id-embeddng-generation-with-diffusion)
    - [Personalized Content Generation with Diffusion](#personalized-content-generation-with-diffusion)
- [Resources](#resources)
    - [Tutorials](#tutorials)
    - [Talks](#talks)
    - [Courses](#courses)
    - [Open Source Projects](#open-source-projects)
    - [Workshops](#workshops)


## Papers

### Overview & Surveys

### LLM-based Generative Recommendation

#### Zero-shot Recommendation with LLMs

1. **Large language models are zero-shot rankers for recommender systems.** ECIR 2024. [[paper](https://arxiv.org/pdf/2305.08845)]

   *Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, Wayne Xin Zhao.*

2. **Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System.** Arxiv 2023. [[paper](https://arxiv.org/pdf/2303.14524)]

   *Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, Jiawei Zhang.*

3. **Is ChatGPT a Good Recommender? A Preliminary Study.** CIKM 2023. [[paper](https://arxiv.org/pdf/2304.10149)]

   *Junling Liu, Chao Liu, Peilin Zhou, Renjie Lv, Kang Zhou, Yan Zhang.*

4. **Uncovering ChatGPT's Capabilities in Recommender Systems.** RecSys 2023. [[paper](https://arxiv.org/abs/2305.02182)]

   *Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, Jun Xu.*

5. **Leveraging Large Language Models for Sequential Recommendation.** RecSys 2023. [[paper](https://arxiv.org/pdf/2309.09261)]

   *Jesse Harte, Wouter Zorgdrager, Panos Louridas, Asterios Katsifodimos, Dietmar Jannach, Marios Fragkoulis.*

#### Aligning LLMs with User Behaviors

1. **TallRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation.** RecSys 2023. [[paper](https://arxiv.org/pdf/2305.00447)]

   *Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, Xiangnan He.*

2. **On Softmax Direct Preference Optimization for Recommendation.** NeurIPS 2024. [[paper](https://arxiv.org/pdf/2406.09215)]

   *Yuxin Chen, Junfei Tan, An Zhang, Zhengyi Yang, Leheng Sheng, Enzhi Zhang, Xiang Wang, Tat-Seng Chua.*


3. **LLaRA: Large Language-Recommendation Assistant.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2312.02445)]

   *Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, Xiang Wang, Xiangnan He.*



#### LLM-powered Agents in Recommendation
1. **On Generative Agents in Recommendation.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2310.10108)]

   *An Zhang, Yuxin Chen, Leheng Sheng, Xiang Wang, Tat-Seng Chua.*

2. **Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations.** Arxiv 2024. [[paper](https://arxiv.org/pdf/2308.16505)]

   *Xu Huang, Jianxun Lian, Yuxuan Lei, Jing Yao, Defu Lian, Xing Xie.*


3. **RecMind: Large Language Model Powered Agent For Recommendation.** Arxiv 2024. [[paper](https://arxiv.org/pdf/2308.14296)]

   *Yancheng Wang, Ziyan Jiang, Zheng Chen, Fan Yang, Yingxue Zhou, Eunah Cho, Xing Fan, Xiaojiang Huang, Yanbin Lu, Yingzhen Yang.*


4. **RAH! RecSys–Assistant–Human: A Human-Centered Recommendation Framework With LLM Agents.** TOCS 2024. [[paper](https://arxiv.org/pdf/2308.09904)]

   *Yubo Shu, Haonan Zhang, Hansu Gu, Peng Zhang, Tun Lu, Dongsheng Li, Ning Gu.*


5. **MACRec: A Multi-Agent Collaboration Framework for Recommendation.** SIGIR 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3626772.3657669)]

   *Zhefan Wang, Yuanqing Yu, Wendi Zheng, Weizhi Ma, Min Zhang.*


6. **Let Me Do It For You: Towards LLM Empowered Recommendation via Tool Learning.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2405.15114)]

   *Yuyue Zhao, Jiancan Wu, Xiang Wang, Wei Tang, Dingxian Wang, Maarten de Rijke.*

7. **User Behavior Simulation with Large Language Model-based Agents for Recommender Systems.** TOIS 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3708985)]

   *Lei Wang, Jingsen Zhang, Hao Yang, Zhi-Yuan Chen, Jiakai Tang, Zeyu Zhang, Xu Chen, Yankai Lin, Hao Sun, Ruihua Song, Wayne Xin Zhao, Jun Xu, Zhicheng Dou, Jun Wang, Ji-Rong Wen.*

8. **Bridging Items and Language: A Transition Paradigm for Large Language Model-Based Recommendation.** KDD 2024. [[paper](https://arxiv.org/pdf/2310.06491)]

   *Xinyu Lin, Wenjie Wang, Yongqi Li, Fuli Feng, See-Kiong Ng, Tat-Seng Chua.*

9. **A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems.** Arxiv 2023. [[paper](https://arxiv.org/pdf/2308.08434)]

   *Keqin Bao, Jizhi Zhang, Wenjie Wang, Yang Zhang, Zhengyi Yang, Yancheng Luo, Chong Chen, Fuli Feng, Qi Tian.*


10. **Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation.** EMNLP 2024. [[paper](https://arxiv.org/pdf/2406.14900)]

   *Keqin Bao, Jizhi Zhang, Yang Zhang, Xinyue Huo, Chong Chen, Fuli Feng.*

11. **Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach.** TOIS 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3708882)]

   *Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, Ji-Rong Wen.*

12. **Generative Recommendation: Towards Next-generation Recommender Paradigm.** Arxiv 2023. [[paper](https://arxiv.org/pdf/2304.03516)]

    *Wenjie Wang, Xinyu Lin, Fuli Feng, Xiangnan He, Tat-Seng Chua.*

13. **SLMREC: Empowering Small Language Models for Sequential Recommendation.** Arxiv 2024. [[paper](https://arxiv.org/pdf/2405.17890)]

   *Wujiang Xu, Qitian Wu, Zujie Liang, Jiaojiao Han, Xuying Ning, Yunxiao Shi, Wenfang Lin, Yongfeng Zhang.*


14. **Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5).** RecSys 2022. [[paper](https://arxiv.org/pdf/2203.13366)]

   *Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, Yongfeng Zhang.*


15. **M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems.** Arxiv 2022. [[paper](https://arxiv.org/pdf/2205.08084)]

   *Zeyu Cui, Jianxin Ma, Chang Zhou, Jingren Zhou, Hongxia Yang.*


16. **Data-efficient Fine-tuning for LLM-based Recommendation.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2401.17197)]

    *Xinyu Lin, Wenjie Wang, Yongqi Li, Shuo Yang, Fuli Feng, Yinwei Wei, Tat-Seng Chua.*

17. **Collaborative Large Language Model for Recommender Systems.** WWW 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3589334.3645347)]

    *Yaochen Zhu, Liang Wu, Qi Guo, Liangjie Hong, Jundong Li.*

#### LLM-based Conversational Recommender Systems

1. **LLM-REDIAL: A Large-Scale Dataset for Conversational Recommender Systems Created from User Behaviors with LLMs.** ACL Findings 2024. [[paper](https://aclanthology.org/2024.findings-acl.529.pdf)]

   *Tingting Liang, Chenxin Jin, Lingzhi Wang, Wenqi Fan, Congying Xia, Kai Chen, Yuyu Yin.*


2. **Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models.** EMNLP 2023. [[paper](https://arxiv.org/pdf/2305.13112)]

   *Xiaolei Wang, Xinyu Tang, Wayne Xin Zhao, Jingyuan Wang, Ji-Rong Wen.*


3. **How Reliable is Your Simulator? Analysis on the Limitations of Current LLM-based User Simulators for Conversational Recommendation.** WWW 2024. [[paper](https://arxiv.org/pdf/2403.16416)]

   *Lixi Zhu, Xiaowen Huang, Jitao Sang.*


4. **Large Language Models as Zero-Shot Conversational Recommenders.** CIKM 2023. [[paper](https://dl.acm.org/doi/pdf/10.1145/3583780.3614949)]

   *Zhankui He, Zhouhang Xie, Rahul Jha, Harald Steck, Dawen Liang, Yesu Feng, Bodhisattwa Prasad Majumder, Nathan Kallus, Julian McAuley.*



### Semantic ID-based Generative Recommendation

#### Generative Retrieval and Semantic IDs

#### Semantic ID-based Recommender Architecture

1. **Recommender Systems with Generative Retrieval.** NeurIPS 2023. [[paper](https://arxiv.org/abs/2305.05065)]

   *Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy.*

#### Semantic ID Construction

### Diffusion Model-based Generative Recommendation

#### Diffusion-based Recommendation Architecture

#### ID Embedding Generation with Diffusion

#### Personalized Content Generation with Diffusion

## Resouces

### Tutorials

### Talks

### Courses

### Open Source Projects

* [Awesome-Generative-RecSys](https://github.com/jihoo-kim/Awesome-Generative-RecSys) - A repo featuring papers on generative recommender systems, though not actively maintained.

### Workshops
