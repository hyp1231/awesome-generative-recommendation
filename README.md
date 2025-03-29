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

* (LLMRank) **Large language models are zero-shot rankers for recommender systems.** ECIR 2024. [[paper](https://arxiv.org/pdf/2305.08845)] [[code](https://github.com/RUCAIBox/LLMRank)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/LLMRank)


   *Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, Wayne Xin Zhao.*

* (Chat-REC) **Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System.** arXiv:2303.14524. [[paper](https://arxiv.org/pdf/2303.14524)] 

   *Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, Jiawei Zhang.*

* **Is ChatGPT a Good Recommender? A Preliminary Study.** CIKM 2023. [[paper](https://arxiv.org/pdf/2304.10149)] [[code](https://github.com/williamliujl/LLMRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/williamliujl/LLMRec)

   *Junling Liu, Chao Liu, Peilin Zhou, Renjie Lv, Kang Zhou, Yan Zhang.*

* **Uncovering ChatGPT's Capabilities in Recommender Systems.** RecSys 2023. [[paper](https://arxiv.org/pdf/2305.02182)] [[code](https://github.com/rainym00d/LLM4RS)] ![GitHub Repo stars](https://img.shields.io/github/stars/rainym00d/LLM4RS)

   *Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, Jun Xu.*

* **Leveraging Large Language Models for Sequential Recommendation.** RecSys 2023. [[paper](https://arxiv.org/pdf/2309.09261)] [[code](https://github.com/dh-r/LLM-Sequential-Recommendation)] ![GitHub Repo stars](https://img.shields.io/github/stars/dh-r/LLM-Sequential-Recommendation)

   *Jesse Harte, Wouter Zorgdrager, Panos Louridas, Asterios Katsifodimos, Dietmar Jannach, Marios Fragkoulis.*

#### Aligning LLMs with User Behaviors

* (TallRec) **TallRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation.** RecSys 2023. [[paper](https://arxiv.org/pdf/2305.00447)] [[code](https://github.com/SAI990323/TALLRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/SAI990323/TALLRec)

   *Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, Xiangnan He.*

* (S-DPO) **On Softmax Direct Preference Optimization for Recommendation.** NeurIPS 2024. [[paper](https://arxiv.org/pdf/2406.09215)] [[code](https://github.com/chenyuxin1999/S-DPO)] ![GitHub Repo stars](https://img.shields.io/github/stars/chenyuxin1999/S-DPO)

   *Yuxin Chen, Junfei Tan, An Zhang, Zhengyi Yang, Leheng Sheng, Enzhi Zhang, Xiang Wang, Tat-Seng Chua.*


* (LLaRA) **LLaRA: Large Language-Recommendation Assistant.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2312.02445)] [[code](https://github.com/ljy0ustc/LLaRA)] ![GitHub Repo stars](https://img.shields.io/github/stars/ljy0ustc/LLaRA)

   *Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, Xiang Wang, Xiangnan He.*

* (I-LLMRec) **Image is All You Need: Towards Efficient and Effective Large Language Model-Based Recommender Systems.** arXiv:2503.06238. [[paper](https://arxiv.org/pdf/2503.06238)] [[code](https://github.com/rlqja1107/torch-I-LLMRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/rlqja1107/torch-I-LLMRec)

   *Kibum Kim, Sein Kim, Hongseok Kang, Jiwan Kim, Heewoong Noh, Yeonjun In, Kanghoon Yoon, Jinoh Oh, Chanyoung Park.*


#### LLM-powered Agents in Recommendation
* (Agent4Rec) **On Generative Agents in Recommendation.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2310.10108)] [[code](https://github.com/LehengTHU/Agent4Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/LehengTHU/Agent4Rec)

   *An Zhang, Yuxin Chen, Leheng Sheng, Xiang Wang, Tat-Seng Chua.*

* (InteRecAgent) **Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations.** arXiv:2308.16505. [[paper](https://arxiv.org/pdf/2308.16505)] [[code](https://github.com/microsoft/RecAI)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/RecAI)

   *Xu Huang, Jianxun Lian, Yuxuan Lei, Jing Yao, Defu Lian, Xing Xie.*


* (RecMind) **RecMind: Large Language Model Powered Agent For Recommendation.** NACCL 2024 (Findings). [[paper](https://arxiv.org/pdf/2308.14296)]

   *Yancheng Wang, Ziyan Jiang, Zheng Chen, Fan Yang, Yingxue Zhou, Eunah Cho, Xing Fan, Xiaojiang Huang, Yanbin Lu, Yingzhen Yang.*


* (RAH) **RAH! RecSys–Assistant–Human: A Human-Centered Recommendation Framework With LLM Agents.** TOCS 2024. [[paper](https://arxiv.org/pdf/2308.09904)]

   *Yubo Shu, Haonan Zhang, Hansu Gu, Peng Zhang, Tun Lu, Dongsheng Li, Ning Gu.*


* (MACRec) **MACRec: A Multi-Agent Collaboration Framework for Recommendation.** SIGIR 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3626772.3657669)] [[code](https://github.com/wzf2000/MACRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/wzf2000/MACRec)

   *Zhefan Wang, Yuanqing Yu, Wendi Zheng, Weizhi Ma, Min Zhang.*


* (ToolRec) **Let Me Do It For You: Towards LLM Empowered Recommendation via Tool Learning.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2405.15114)]

   *Yuyue Zhao, Jiancan Wu, Xiang Wang, Wei Tang, Dingxian Wang, Maarten de Rijke.*

* (RecAgent) **User Behavior Simulation with Large Language Model-based Agents for Recommender Systems.** TOIS 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3708985)] [[code](https://github.com/RUC-GSAI/YuLan-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUC-GSAI/YuLan-Rec)

   *Lei Wang, Jingsen Zhang, Hao Yang, Zhi-Yuan Chen, Jiakai Tang, Zeyu Zhang, Xu Chen, Yankai Lin, Hao Sun, Ruihua Song, Wayne Xin Zhao, Jun Xu, Zhicheng Dou, Jun Wang, Ji-Rong Wen.*

* (TransRec) **Bridging Items and Language: A Transition Paradigm for Large Language Model-Based Recommendation.** KDD 2024. [[paper](https://arxiv.org/pdf/2310.06491)] [[code](https://github.com/Linxyhaha/TransRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/TransRec)

   *Xinyu Lin, Wenjie Wang, Yongqi Li, Fuli Feng, See-Kiong Ng, Tat-Seng Chua.*

* (BIGRec) **A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems.** arXiv:2308.08434. [[paper](https://arxiv.org/pdf/2308.08434)] [[code](https://github.com/SAI990323/BIGRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/SAI990323/BIGRec)

   *Keqin Bao, Jizhi Zhang, Wenjie Wang, Yang Zhang, Zhengyi Yang, Yancheng Luo, Chong Chen, Fuli Feng, Qi Tian.*


* (D3) **Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation.** EMNLP 2024. [[paper](https://arxiv.org/pdf/2406.14900)] [[code](https://github.com/SAI990323/DecodingMatters)] ![GitHub Repo stars](https://img.shields.io/github/stars/SAI990323/DecodingMatters)

   *Keqin Bao, Jizhi Zhang, Yang Zhang, Xinyue Huo, Chong Chen, Fuli Feng.*

* (InstructRec) **Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach.** TOIS 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3708882)]

   *Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, Ji-Rong Wen.*

* (GenRec) **Generative Recommendation: Towards Next-generation Recommender Paradigm.** arXiv:2304.03516. [[paper](https://arxiv.org/pdf/2304.03516)] [[code](https://github.com/Linxyhaha/GeneRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/GeneRec)

    *Wenjie Wang, Xinyu Lin, Fuli Feng, Xiangnan He, Tat-Seng Chua.*

* (SLMREC) **SLMREC: Empowering Small Language Models for Sequential Recommendation.** arXiv:2405.17890. [[paper](https://arxiv.org/pdf/2405.17890)] [[code](https://github.com/WujiangXu/SLMRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/WujiangXu/SLMRec)

   *Wujiang Xu, Qitian Wu, Zujie Liang, Jiaojiao Han, Xuying Ning, Yunxiao Shi, Wenfang Lin, Yongfeng Zhang.*


* (P5) **Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5).** RecSys 2022. [[paper](https://arxiv.org/pdf/2203.13366)] [[code](https://github.com/jeykigung/P5)] ![GitHub Repo stars](https://img.shields.io/github/stars/jeykigung/P5)

   *Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, Yongfeng Zhang.*


* (M6-Rec) **M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems.** arXiv:2205.08084. [[paper](https://arxiv.org/pdf/2205.08084)]

   *Zeyu Cui, Jianxin Ma, Chang Zhou, Jingren Zhou, Hongxia Yang.*


* (DEALRec) **Data-efficient Fine-tuning for LLM-based Recommendation.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2401.17197)] [[code](https://github.com/Linxyhaha/DEALRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/DEALRec)

    *Xinyu Lin, Wenjie Wang, Yongqi Li, Shuo Yang, Fuli Feng, Yinwei Wei, Tat-Seng Chua.*

* (CLLM4Rec) **Collaborative Large Language Model for Recommender Systems.** WWW 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3589334.3645347)] [[code](https://github.com/yaochenzhu/llm4rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/yaochenzhu/llm4rec)

    *Yaochen Zhu, Liang Wu, Qi Guo, Liangjie Hong, Jundong Li.*


* (RecExplainer) **RecExplainer: Aligning Large Language Models for Explaining Recommendation Models.** KDD 2024. [[paper](https://arxiv.org/pdf/2311.10947)] [[code](https://github.com/microsoft/RecAI)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/RecAI)

   *Yuxuan Lei, Jianxun Lian, Jing Yao, Xu Huang, Defu Lian, Xing Xie.*

* (AgentCF) **AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems.** WWW 2024. [[paper](https://arxiv.org/pdf/2310.09233)]

   *Junjie Zhang, Yupeng Hou, Ruobing Xie, Wenqi Sun, Julian McAuley, Wayne Xin Zhao, Leyu Lin, Ji-Rong Wen.*

* (P5-ID)**How to Index Item IDs for Recommendation Foundation Models.** SIGIR-AP 2023. [[paper](https://dl.acm.org/doi/pdf/10.1145/3624918.3625339)] [[code](https://github.com/Wenyueh/LLM-RecSys-ID)] ![GitHub Repo stars](https://img.shields.io/github/stars/Wenyueh/LLM-RecSys-ID)

    *Wenyue Hua, Shuyuan Xu, Yingqiang Ge, Yongfeng Zhang.*

* (ReLLa) **ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation.** arXiv:2308.11131. [[paper](https://arxiv.org/pdf/2308.11131)] [[code](https://github.com/LaVieEnRose365/ReLLa)] ![GitHub Repo stars](https://img.shields.io/github/stars/LaVieEnRose365/ReLLa)

    *Jianghao Lin, Rong Shan, Chenxu Zhu, Kounianhua Du, Bo Chen, Shigang Quan, Ruiming Tang, Yong Yu, and Weinan Zhang.*

* (LC-Rec) **Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation.** ICDE 2024. [[paper](https://arxiv.org/pdf/2311.09049)] [[code](https://github.com/RUCAIBox/LC-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/LC-Rec)

    *Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, Ji-Rong Wen.*


* (Collm) **Collm: Integrating collaborative embeddings into large language models for recommendation.** arXiv preprint arXiv:2310.19488. [[paper](https://arxiv.org/pdf/2310.19488)] [[code](https://github.com/zyang1580/CoLLM)] ![GitHub Repo stars](https://img.shields.io/github/stars/zyang1580/CoLLM)

    *Yang Zhang, Fuli Feng, Jizhi Zhang, Keqin Bao, Qifan Wang and Xiangnan He.*

* (E4SRec) **E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation.** arXiv:2312.02443. [[paper](https://arxiv.org/pdf/2312.02443)] [[code](https://github.com/HestiaSky/E4SRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/HestiaSky/E4SRec)

    *Xinhang Li, Chong Chen, Xiangyu Zhao, Yong Zhang, Chunxiao Xing.*

* (Recformer) **Text Is All You Need: Learning Language Representations for Sequential Recommendation.** KDD 2023. [[paper](https://arxiv.org/pdf/2305.13731)]

    *Jiacheng Li, Ming Wang, Jin Li, Jinmiao Fu, Xin Shen, Jingbo Shang, and Julian McAuley.*

* (GenRec) **GenRec: Large Language Model for Generative Recommendation.** ECIR 2024. [[paper](https://openreview.net/pdf?id=KiX8CW0bCr)] [[code](https://github.com/rutgerswiselab/GenRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/rutgerswiselab/GenRec)

    *Jianchao Ji, Zelong Li, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Juntao Tan, Yongfeng Zhang.*


* (ONCE) **ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models.** WSDM 2024. [[paper](https://arxiv.org/pdf/2305.06566)] [[code](https://github.com/Jyonn/ONCE)] ![GitHub Repo stars](https://img.shields.io/github/stars/Jyonn/ONCE)

   *Qijiong Liu, Nuo Chen, Tetsuya Sakai, Xiao-Ming Wu.*

* (HKFR) **Heterogeneous Knowledge Fusion: A Novel Approach for Personalized Recommendation via LLM.** RecSys 2023. [[paper](https://arxiv.org/pdf/2308.03333)]
   *Bin Yin, Junjie Xie, Yu Qin, Zixiang Ding, Zhichao Feng, Xiang Li, Wei Lin.*

* (LlamaRec) **LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking.** PGAI@CIKM 2023. [[paper](https://arxiv.org/pdf/2311.02089)] [[code](https://github.com/Yueeeeeeee/LlamaRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Yueeeeeeee/LlamaRec)
   *Zhenrui Yue, Sara Rabhi, Gabriel de Souza Pereira Moreira, Dong Wang, Even Oldridge.*





#### LLM-based Conversational Recommender Systems

* (LLM-REDIAL) **LLM-REDIAL: A Large-Scale Dataset for Conversational Recommender Systems Created from User Behaviors with LLMs.** ACL Findings 2024. [[paper](https://aclanthology.org/2024.findings-acl.529.pdf)] [[code](https://github.com/LitGreenhand/LLM-Redial)] ![GitHub Repo stars](https://img.shields.io/github/stars/LitGreenhand/LLM-Redial)

   *Tingting Liang, Chenxin Jin, Lingzhi Wang, Wenqi Fan, Congying Xia, Kai Chen, Yuyu Yin.*


* (iEvaLM) **Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models.** EMNLP 2023. [[paper](https://arxiv.org/pdf/2305.13112)] [[code](https://github.com/RUCAIBox/iEvaLM-CRS)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/iEvaLM-CRS)

   *Xiaolei Wang, Xinyu Tang, Wayne Xin Zhao, Jingyuan Wang, Ji-Rong Wen.*


*  **How Reliable is Your Simulator? Analysis on the Limitations of Current LLM-based User Simulators for Conversational Recommendation.** WWW 2024. [[paper](https://arxiv.org/pdf/2403.16416)] [[code](https://github.com/RUCAIBox/iEvaLM-CRS)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/iEvaLM-CRS)

   *Lixi Zhu, Xiaowen Huang, Jitao Sang.*


* **Large Language Models as Zero-Shot Conversational Recommenders.** CIKM 2023. [[paper](https://dl.acm.org/doi/pdf/10.1145/3583780.3614949)] [[code](https://github.com/AaronHeee/LLMs-as-Zero-Shot-Conversational-RecSys)] ![GitHub Repo stars](https://img.shields.io/github/stars/AaronHeee/LLMs-as-Zero-Shot-Conversational-RecSys)

   *Zhankui He, Zhouhang Xie, Rahul Jha, Harald Steck, Dawen Liang, Yesu Feng, Bodhisattwa Prasad Majumder, Nathan Kallus, Julian McAuley.*



### Semantic ID-based Generative Recommendation

#### SemID-based Generative Recommender Architecture

* (TIGER) **Recommender Systems with Generative Retrieval.** NeurIPS 2023. [[paper](https://arxiv.org/abs/2305.05065)]

   *Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy.*

* (HSTU) **Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations.** ICML 2024. [[paper](https://arxiv.org/abs/2402.17152)] [[code](https://github.com/facebookresearch/generative-recommenders)] ![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/generative-recommenders)

    *Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, Yinghai Lu, Yu Shi.*

* **EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration.** KDD 2024. [[paper](https://arxiv.org/abs/2406.14017)] [[code](https://github.com/yewzz/EAGER)] ![GitHub Repo stars](https://img.shields.io/github/stars/yewzz/EAGER)

    *Ye Wang, Jiahao Xun, Minjie Hong, Jieming Zhu, Tao Jin, Wang Lin, Haoyuan Li, Linjun Li, Yan Xia, Zhou Zhao, Zhenhua Dong.*

* **SC-Rec: Enhancing Generative Retrieval with Self-Consistent Reranking for Sequential Recommendation.** arXiv:2408.08686. [[paper](https://arxiv.org/abs/2408.08686)]

    *Tongyoung Kim, Soojin Yoon, Seongku Kang, Jinyoung Yeo, Dongha Lee.*

* (SpecGR) **Inductive Generative Recommendation via Retrieval-based Speculation.** arXiv:2410.02939. [[paper](https://arxiv.org/abs/2410.02939)] [[code](https://github.com/Jamesding000/SpecGR)] ![GitHub Repo stars](https://img.shields.io/github/stars/Jamesding000/SpecGR)

    *Yijie Ding, Yupeng Hou, Jiacheng Li, Julian McAuley.*

* (LIGER) **Unifying Generative and Dense Retrieval for Sequential Recommendation.** arXiv:2411.18814. [[paper](https://arxiv.org/abs/2411.18814)]

    *Liu Yang, Fabian Paischer, Kaveh Hassani, Jiacheng Li, Shuai Shao, Zhang Gabriel Li, Yun He, Xue Feng, Nima Noorshams, Sem Park, Bo Long, Robert D Nowak, Xiaoli Gao, Hamid Eghbalzadeh.*

* **OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment.** arXiv:2502.18965. [[paper](https://arxiv.org/abs/2502.18965)]

    *Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, Guorui Zhou.*

#### Item Tokenization

* (DSI) **Transformer Memory as a Differentiable Search Index.** NeurIPS 2022. [[paper](https://arxiv.org/abs/2202.06991)]

   *Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, Donald Metzler.*

* (NCI) **A Neural Corpus Indexer for Document Retrieval.** NeurIPS 2022. [[paper](https://arxiv.org/abs/2206.02743)] [[code](https://github.com/solidsea98/Neural-Corpus-Indexer-NCI)] ![GitHub Repo stars](https://img.shields.io/github/stars/solidsea98/Neural-Corpus-Indexer-NCI)

   *Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Hao Sun, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie, Hao Allen Sun, Weiwei Deng, Qi Zhang, Mao Yang.*

* (VQ-Rec) **Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders.** WWW 2023. [[paper](https://arxiv.org/abs/2210.12316)] [[code](https://github.com/RUCAIBox/VQ-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/VQ-Rec)

   *Yupeng Hou, Zhankui He, Julian McAuley, Wayne Xin Zhao.*

* **How to Index Item IDs for Recommendation Foundation Models.** SIGIR-AP 2023. [[paper](https://arxiv.org/abs/2305.06569)] [[code](https://github.com/Wenyueh/LLM-RecSys-ID)] ![GitHub Repo stars](https://img.shields.io/github/stars/Wenyueh/LLM-RecSys-ID)

   *Wenyue Hua, Shuyuan Xu, Yingqiang Ge, Yongfeng Zhang.*

* **Generative Sequential Recommendation with GPTRec.** Gen-IR @ SIGIR 2023 workshop. [[paper](https://arxiv.org/abs/2306.11114)]

   *Aleksandr V. Petrov, Craig Macdonald.*

* (SEATER) **Generative Retrieval with Semantic Tree-Structured Item Identifiers via Contrastive Learning.** SIGIR-AP 2024. [[paper](https://arxiv.org/abs/2309.13375)] [[code](https://github.com/ethan00si/seater_generative_retrieval)] ![GitHub Repo stars](https://img.shields.io/github/stars/ethan00si/seater_generative_retrieval)

    *Zihua Si, Zhongxiang Sun, Jiale Chen, Guozhang Chen, Xiaoxue Zang, Kai Zheng, Yang Song, Xiao Zhang, Jun Xu, Kun Gai.*

* (ColaRec) **Content-Based Collaborative Generation for Recommender Systems.** CIKM 2024. [[paper](https://arxiv.org/abs/2403.18480)] [[code](https://github.com/Junewang0614/ColaRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Junewang0614/ColaRec)

    *Yidan Wang, Zhaochun Ren, Weiwei Sun, Jiyuan Yang, Zhixiang Liang, Xin Chen, Ruobing Xie, Su Yan, Xu Zhang, Pengjie Ren, Zhumin Chen, Xin Xin.*

* **CoST: Contrastive Quantization based Semantic Tokenization for Generative Recommendation.** RecSys 2024. [[paper](https://arxiv.org/abs/2404.14774)]

    *Jieming Zhu, Mengqun Jin, Qijiong Liu, Zexuan Qiu, Zhenhua Dong, Xiu Li.*

* **MMGRec: Multimodal Generative Recommendation with Transformer Model.** arXiv:2404.16555. [[paper](https://arxiv.org/abs/2404.16555)]

    *Han Liu, Yinwei Wei, Xuemeng Song, Weili Guan, Yuan-Fang Li, Liqiang Nie.*

* (LETTER) **Learnable Item Tokenization for Generative Recommendation.** CIKM 2024. [[paper](https://arxiv.org/abs/2405.07314)] [[code](https://github.com/HonghuiBao2000/LETTER)] ![GitHub Repo stars](https://img.shields.io/github/stars/HonghuiBao2000/LETTER)

    *Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, See-Kiong Ng, Tat-Seng Chua.*

* (MBGen) **Multi-Behavior Generative Recommendation.** CIKM 2024. [[paper](https://arxiv.org/abs/2405.16871)] [[code](https://github.com/anananan116/MBGen)] ![GitHub Repo stars](https://img.shields.io/github/stars/anananan116/MBGen)

    *Zihan Liu, Yupeng Hou, Julian McAuley.*

* (ETEGRec) **End-to-End Learnable Item Tokenization for Generative Recommendation.** arXiv:2409.05546. [[paper](https://arxiv.org/abs/2409.05546)]

    *Enze Liu, Bowen Zheng, Cheng Ling, Lantao Hu, Han Li, Wayne Xin Zhao.*

* (MoC) **Towards Scalable Semantic Representation for Recommendation.** arXiv:2410.09560. [[paper](https://arxiv.org/abs/2410.09560)]

    *Taolin Zhang, Junwei Pan, Jinpeng Wang, Yaohua Zha, Tao Dai, Bin Chen, Ruisheng Luo, Xiaoxiang Deng, Yuan Wang, Ming Yue, Jie Jiang, Shu-Tao Xia.*

* (PRORec) **Progressive Collaborative and Semantic Knowledge Fusion for Generative Recommendation.** arXiv:2502.06269. [[paper](https://arxiv.org/abs/2502.06269)]

    *Longtao Xiao, Haozhao Wang, Cheng Wang, Linfei Ji, Yifan Wang, Jieming Zhu, Zhenhua Dong, Rui Zhang, Ruixuan Li.*

* **ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation.** arXiv:2502.13581. [[paper](https://arxiv.org/abs/2502.13581)]

    *Yupeng Hou, Jianmo Ni, Zhankui He, Noveen Sachdeva, Wang-Cheng Kang, Ed H. Chi, Julian McAuley, Derek Zhiyuan Cheng.*

#### Aligning with Language Models

* **Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5).** RecSys 2022. [[paper](https://arxiv.org/abs/2203.13366)] [[code](https://github.com/jeykigung/P5)] ![GitHub Repo stars](https://img.shields.io/github/stars/jeykigung/P5)

   *Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, Yongfeng Zhang.*

* (LMIndexer) **Language Models As Semantic Indexers.** ICDE 2024. [[paper](https://arxiv.org/abs/2310.07815)] [[code](https://github.com/PeterGriffinJin/LMIndexer)] ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/LMIndexer)

    *Bowen Jin, Hansi Zeng, Guoyin Wang, Xiusi Chen, Tianxin Wei, Ruirui Li, Zhengyang Wang, Zheng Li, Yang Li, Hanqing Lu, Suhang Wang, Jiawei Han, Xianfeng Tang.*

* (LC-Rec) **Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation.** ICDE 2024. [[paper](https://arxiv.org/abs/2311.09049)] [[code](https://github.com/RUCAIBox/LC-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/LC-Rec)

    *Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, Ji-Rong Wen.*

* **IDGenRec: LLM-RecSys Alignment with Textual ID Learning.** SIGIR 2024. [[paper](https://arxiv.org/abs/2403.19021)] [[code](https://github.com/agiresearch/IDGenRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/agiresearch/IDGenRec)

    *Juntao Tan, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Zelong Li, Yongfeng Zhang.*

* (AtSpeed) **Efficient Inference for Large Language Model-based Generative Recommendation.** ICLR 2025. [[paper](https://arxiv.org/abs/2410.05165)] [[code](https://github.com/Linxyhaha/AtSpeed)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/AtSpeed)

    *Xinyu Lin, Chaoqun Yang, Wenjie Wang, Yongqi Li, Cunxiao Du, Fuli Feng, See-Kiong Ng, Tat-Seng Chua.*

* **Semantic Convergence: Harmonizing Recommender Systems via Two-Stage Alignment and Behavioral Semantic Tokenization.** AAAI 2025. [[paper](https://arxiv.org/abs/2412.13771)]

    *Guanghan Li, Xun Zhang, Yufei Zhang, Yifan Yin, Guojun Yin, Wei Lin.*

* (SETRec) **Order-agnostic Identifier for Large Language Model-based Generative Recommendation.** arXiv:2502.10833. [[paper](https://arxiv.org/abs/2502.10833)]

    *Xinyu Lin, Haihan Shi, Wenjie Wang, Fuli Feng, Qifan Wang, See-Kiong Ng, Tat-Seng Chua.*

* **EAGER-LLM: Enhancing Large Language Models as Recommenders through Exogenous Behavior-Semantic Integration.** WWW 2025. [[paper](https://arxiv.org/abs/2502.14735)]

    *Minjie Hong, Yan Xia, Zehan Wang, Jieming Zhu, Ye Wang, Sihang Cai, Xiaoda Yang, Quanyu Dai, Zhenhua Dong, Zhimeng Zhang, Zhou Zhao.*

### Diffusion Model-based Generative Recommendation

#### Diffusion-enhanced Recommendation

* **Diffusion Augmentation for Sequential Recommendation.** CIKM 2023. [[paper](https://arxiv.org/abs/2309.12858)]

    *Qidong Liu, Fan Yan, Xiangyu Zhao, Zhaocheng Du, Huifeng Guo, Ruiming Tang, Feng Tian.*

* **Diff4Rec: Sequential Recommendation withCurriculum-scheduled Diffusion Augmentation.** MM 2023. [[paper](https://dl.acm.org/doi/10.1145/3581783.3612709)]

    *Zihao Wu, Xin Wang, Hong Chen, Kaidong Li, Yi Han, Lifeng Sun, Wenwu Zhu.*

* **DiffMM: Multi-Modal Diffusion Model for Recommendation.** MM 2024. [[paper](https://arxiv.org/abs/2406.11781)]

    *Yangqin Jiang, Lianghao Xia, Wei Wei, Da Luo, Kangyi Lin, Chao Huang.*

* **Conditional Denoising Diffusion for Sequential Recommendation.** PAKDD 2024. [[paper](https://arxiv.org/abs/2304.11433)]

    *Yu Wang, Zhiwei Liu, Liangwei Yang, Philip S. Yu.*



#### Diffusion as Recommender

* **Diffusion Recommender Model.** SIGIR 2023. [[paper](https://arxiv.org/abs/2304.04971)]

    *Wenjie Wang, Yiyan Xu, Fuli Feng, Xinyu Lin, Xiangnan He, Tat-Seng Chua.*

* **Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion.** NeurIPS 2023. [[paper](https://arxiv.org/abs/2310.20453)]

    *Zhengyi Yang, Jiancan Wu, Zhicai Wang, Xiang Wang, Yancheng Yuan, Xiangnan He.*

* **Bridging User Dynamics: Transforming Sequential Recommendations with Schrödinger Bridge and Diffusion Models.** CIKM 2024. [[paper](https://arxiv.org/abs/2409.10522)]

    *Wenjia Xie, Rui Zhou, Hao Wang, Tingjia Shen, Enhong Chen.*

* **DimeRec: A Unified Framework for Enhanced Sequential Recommendation via Generative Diffusion Models.** CIKM 2024. [[paper](https://arxiv.org/abs/2408.12153)]

    *Wuchao Li, Rui Huang, Haijun Zhao, Chi Liu, Kai Zheng, Qi Liu, Na Mou, Guorui Zhou, Defu Lian, Yang Song, Wentian Bao, Enyun Yu, Wenwu Ou.*

* **Plug-in Diffusion Model for Sequential Recommendation.** AAAI 2024. [[paper](https://arxiv.org/abs/2401.02913)]

    *Haokai Ma, Ruobing Xie, Lei Meng, Xin Chen, Xu Zhang, Leyu Lin, Zhanhui Kang.*

* **SeeDRec: Sememe-based Diffusion for Sequential Recommendation.** IJCAI 2024. [[paper](https://www.ijcai.org/proceedings/2024/251)]

    *Haokai Ma, Ruobing Xie, Lei Meng, Yimeng Yang, Xingwu Sun, Zhanhui Kang.*

* **Breaking Determinism: Fuzzy Modeling of Sequential Recommendation Using Discrete State Space Diffusion Model.** NeurIPS 2024. [[paper](https://arxiv.org/abs/2410.23994)]

    *Wenjia Xie, Hao Wang, Luankang Zhang, Rui Zhou, Defu Lian, Enhong Chen.*

* **Distinguished Quantized Guidance for Diffusion-based Sequence Recommendation.** WWW 2025. [[paper](https://arxiv.org/abs/2501.17670)]

    *Wenyu Mao, Shuchang Liu, Haoyang Liu, Haozhe Liu, Xiang Li, Lantao Hu.*

* **Preference Diffusion for Recommendation.** ICLR 2025. [[paper](https://arxiv.org/abs/2410.13117)]

    *Shuo Liu, An Zhang, Guoqing Hu, Hong Qian, Tat-seng Chua.*


#### Personalized Content Generation with Diffusion

## Resouces

### Tutorials

### Talks

### Courses

### Open Source Projects

* [Awesome-Generative-RecSys](https://github.com/jihoo-kim/Awesome-Generative-RecSys) - A repo featuring papers on generative recommender systems, though not actively maintained.

### Workshops
