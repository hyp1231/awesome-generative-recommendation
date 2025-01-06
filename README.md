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
    - [SemID-based Generative Recommender Architecture](#semid-based-generative-recommender-architecture)
    - [Item Tokenization](#item-tokenization)
    - [Aligning with Language Models](#aligning-with-language-models)
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


18. **RecExplainer: Aligning Large Language Models for Explaining Recommendation Models.** KDD 2024. [[paper](https://arxiv.org/pdf/2311.10947)]

   *Yuxuan Lei, Jianxun Lian, Jing Yao, Xu Huang, Defu Lian, Xing Xie.*

19. **AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems.** WWW 2024. [[paper](https://arxiv.org/pdf/2310.09233)]

   *Junjie Zhang, Yupeng Hou, Ruobing Xie, Wenqi Sun, Julian McAuley, Wayne Xin Zhao, Leyu Lin, Ji-Rong Wen.*

20. **How to Index Item IDs for Recommendation Foundation Models.** SIGIR-AP 2023. [[paper](https://dl.acm.org/doi/pdf/10.1145/3624918.3625339)]

    *Wenyue Hua, Shuyuan Xu, Yingqiang Ge, Yongfeng Zhang.*

21. **ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation.** arXiv preprint arXiv:2308.11131. [[paper](https://arxiv.org/pdf/2308.11131)]

    *Jianghao Lin, Rong Shan, Chenxu Zhu, Kounianhua Du, Bo Chen, Shigang Quan, Ruiming Tang, Yong Yu, and Weinan Zhang.*

22. **Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation.** ICDE 2024. [[paper](https://arxiv.org/pdf/2311.09049)]

    *Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, Ji-Rong Wen.*


23. **Collm: Integrating collaborative embeddings into large language models for recommendation.** arXiv preprint arXiv:2310.19488. [[paper](https://arxiv.org/pdf/2310.19488)]

    *Yang Zhang, Fuli Feng, Jizhi Zhang, Keqin Bao, Qifan Wang and Xiangnan He.*

24. **E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation.** Arxiv 2023. [[paper](https://arxiv.org/pdf/2312.02443)]

    *Xinhang Li, Chong Chen, Xiangyu Zhao, Yong Zhang, Chunxiao Xing.*

25. **Text Is All You Need: Learning Language Representations for Sequential Recommendation.** arXiv preprint arXiv:2305.13731. [[paper](https://arxiv.org/pdf/2305.13731)]

    *Jiacheng Li, Ming Wang, Jin Li, Jinmiao Fu, Xin Shen, Jingbo Shang, and Julian McAuley.*

26. **GenRec: Large Language Model for Generative Recommendation.** ECIR 2024. [[paper](https://openreview.net/pdf?id=KiX8CW0bCr)]

    *Jianchao Ji, Zelong Li, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Juntao Tan, Yongfeng Zhang.*


27. **ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models.** WSDM 2024. [[paper](https://arxiv.org/pdf/2305.06566)]

   *Qijiong Liu, Nuo Chen, Tetsuya Sakai, Xiao-Ming Wu.*

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

#### SemID-based Generative Recommender Architecture

1. (TIGER) **Recommender Systems with Generative Retrieval.** NeurIPS 2023. [[paper](https://arxiv.org/abs/2305.05065)]

   *Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy.*

1. (HSTU) **Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations.** ICML 2024. [[paper](https://arxiv.org/abs/2402.17152)] [[code](https://github.com/facebookresearch/generative-recommenders)] ![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/generative-recommenders)

    *Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, Yinghai Lu, Yu Shi.*

1. **EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration.** KDD 2024. [[paper](https://arxiv.org/abs/2406.14017)] [[code](https://github.com/yewzz/EAGER)] ![GitHub Repo stars](https://img.shields.io/github/stars/yewzz/EAGER)

    *Ye Wang, Jiahao Xun, Minjie Hong, Jieming Zhu, Tao Jin, Wang Lin, Haoyuan Li, Linjun Li, Yan Xia, Zhou Zhao, Zhenhua Dong.*

1. **SC-Rec: Enhancing Generative Retrieval with Self-Consistent Reranking for Sequential Recommendation.** arXiv. [[paper](https://arxiv.org/abs/2408.08686)]

    *Tongyoung Kim, Soojin Yoon, Seongku Kang, Jinyoung Yeo, Dongha Lee.*

1. (SpecGR) **Inductive Generative Recommendation via Retrieval-based Speculation.** arXiv. [[paper](https://arxiv.org/abs/2410.02939)] [[code](https://github.com/Jamesding000/SpecGR)] ![GitHub Repo stars](https://img.shields.io/github/stars/Jamesding000/SpecGR)

    *Yijie Ding, Yupeng Hou, Jiacheng Li, Julian McAuley.*

1. (LIGER) **Unifying Generative and Dense Retrieval for Sequential Recommendation.** arXiv. [[paper](https://arxiv.org/abs/2411.18814)]

    *Liu Yang, Fabian Paischer, Kaveh Hassani, Jiacheng Li, Shuai Shao, Zhang Gabriel Li, Yun He, Xue Feng, Nima Noorshams, Sem Park, Bo Long, Robert D Nowak, Xiaoli Gao, Hamid Eghbalzadeh.*

#### Item Tokenization

1. (DSI) **Transformer Memory as a Differentiable Search Index.** NeurIPS 2022. [[paper](https://arxiv.org/abs/2202.06991)]

   *Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, Donald Metzler.*

1. (NCI) **A Neural Corpus Indexer for Document Retrieval.** NeurIPS 2022. [[paper](https://arxiv.org/abs/2206.02743)] [[code](https://github.com/solidsea98/Neural-Corpus-Indexer-NCI)] ![GitHub Repo stars](https://img.shields.io/github/stars/solidsea98/Neural-Corpus-Indexer-NCI)

   *Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Hao Sun, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie, Hao Allen Sun, Weiwei Deng, Qi Zhang, Mao Yang.*

1. (VQ-Rec) **Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders.** WWW 2023. [[paper](https://arxiv.org/abs/2210.12316)] [[code](https://github.com/RUCAIBox/VQ-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/VQ-Rec)

   *Yupeng Hou, Zhankui He, Julian McAuley, Wayne Xin Zhao.*

1. **How to Index Item IDs for Recommendation Foundation Models.** SIGIR-AP 2023. [[paper](https://arxiv.org/abs/2305.06569)] [[code](https://github.com/Wenyueh/LLM-RecSys-ID)] ![GitHub Repo stars](https://img.shields.io/github/stars/Wenyueh/LLM-RecSys-ID)

   *Wenyue Hua, Shuyuan Xu, Yingqiang Ge, Yongfeng Zhang.*

1. **Generative Sequential Recommendation with GPTRec.** Gen-IR @ SIGIR 2023 workshop. [[paper](https://arxiv.org/abs/2306.11114)]

   *Aleksandr V. Petrov, Craig Macdonald.*

1. (SEATER) **Generative Retrieval with Semantic Tree-Structured Item Identifiers via Contrastive Learning.** SIGIR-AP 2024. [[paper](https://arxiv.org/abs/2309.13375)] [[code](https://github.com/ethan00si/seater_generative_retrieval)] ![GitHub Repo stars](https://img.shields.io/github/stars/ethan00si/seater_generative_retrieval)

    *Zihua Si, Zhongxiang Sun, Jiale Chen, Guozhang Chen, Xiaoxue Zang, Kai Zheng, Yang Song, Xiao Zhang, Jun Xu, Kun Gai.*

1. (ColaRec) **Content-Based Collaborative Generation for Recommender Systems.** CIKM 2024. [[paper](https://arxiv.org/abs/2403.18480)] [[code](https://github.com/Junewang0614/ColaRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Junewang0614/ColaRec)

    *Yidan Wang, Zhaochun Ren, Weiwei Sun, Jiyuan Yang, Zhixiang Liang, Xin Chen, Ruobing Xie, Su Yan, Xu Zhang, Pengjie Ren, Zhumin Chen, Xin Xin.*

1. **CoST: Contrastive Quantization based Semantic Tokenization for Generative Recommendation.** RecSys 2024. [[paper](https://arxiv.org/abs/2404.14774)]

    *Jieming Zhu, Mengqun Jin, Qijiong Liu, Zexuan Qiu, Zhenhua Dong, Xiu Li.*

1. **MMGRec: Multimodal Generative Recommendation with Transformer Model.** arXiv. [[paper](https://arxiv.org/abs/2404.16555)]

    *Han Liu, Yinwei Wei, Xuemeng Song, Weili Guan, Yuan-Fang Li, Liqiang Nie.*

1. (LETTER) **Learnable Item Tokenization for Generative Recommendation.** CIKM 2024. [[paper](https://arxiv.org/abs/2405.07314)] [[code](https://github.com/HonghuiBao2000/LETTER)] ![GitHub Repo stars](https://img.shields.io/github/stars/HonghuiBao2000/LETTER)

    *Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, See-Kiong Ng, Tat-Seng Chua.*

1. (MBGen) **Multi-Behavior Generative Recommendation.** CIKM 2024. [[paper](https://arxiv.org/abs/2405.16871)] [[code](https://github.com/anananan116/MBGen)] ![GitHub Repo stars](https://img.shields.io/github/stars/anananan116/MBGen)

    *Zihan Liu, Yupeng Hou, Julian McAuley.*

1. (ETEGRec) **End-to-End Learnable Item Tokenization for Generative Recommendation.** arXiv. [[paper](https://arxiv.org/abs/2409.05546)]

    *Enze Liu, Bowen Zheng, Cheng Ling, Lantao Hu, Han Li, Wayne Xin Zhao.*

1. (MoC) **Towards Scalable Semantic Representation for Recommendation.** arXiv. [[paper](https://arxiv.org/abs/2410.09560)]

    *Taolin Zhang, Junwei Pan, Jinpeng Wang, Yaohua Zha, Tao Dai, Bin Chen, Ruisheng Luo, Xiaoxiang Deng, Yuan Wang, Ming Yue, Jie Jiang, Shu-Tao Xia.*

#### Aligning with Language Models

1. **Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5).** RecSys 2022. [[paper](https://arxiv.org/abs/2203.13366)] [[code](https://github.com/jeykigung/P5)] ![GitHub Repo stars](https://img.shields.io/github/stars/jeykigung/P5)

   *Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, Yongfeng Zhang.*

1. (LMIndexer) **Language Models As Semantic Indexers.** ICDE 2024. [[paper](https://arxiv.org/abs/2310.07815)] [[code](https://github.com/PeterGriffinJin/LMIndexer)] ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/LMIndexer)

    *Bowen Jin, Hansi Zeng, Guoyin Wang, Xiusi Chen, Tianxin Wei, Ruirui Li, Zhengyang Wang, Zheng Li, Yang Li, Hanqing Lu, Suhang Wang, Jiawei Han, Xianfeng Tang.*

1. (LC-Rec) **Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation.** ICDE 2024. [[paper](https://arxiv.org/abs/2311.09049)] [[code](https://github.com/RUCAIBox/LC-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/LC-Rec)

    *Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, Ji-Rong Wen.*

1. **IDGenRec: LLM-RecSys Alignment with Textual ID Learning.** SIGIR 2024. [[paper](https://arxiv.org/abs/2403.19021)] [[code](https://github.com/agiresearch/IDGenRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/agiresearch/IDGenRec)

    *Juntao Tan, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Zelong Li, Yongfeng Zhang.*

1. (AtSpeed) **Efficient Inference for Large Language Model-based Generative Recommendation.** arXiv. [[paper](https://arxiv.org/abs/2410.05165)]

    *Xinyu Lin, Chaoqun Yang, Wenjie Wang, Yongqi Li, Cunxiao Du, Fuli Feng, See-Kiong Ng, Tat-Seng Chua.*

1. **Semantic Convergence: Harmonizing Recommender Systems via Two-Stage Alignment and Behavioral Semantic Tokenization.** AAAI 2025. [[paper](https://arxiv.org/abs/2412.13771)]

    *Guanghan Li, Xun Zhang, Yufei Zhang, Yifan Yin, Guojun Yin, Wei Lin.*

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
