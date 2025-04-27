# Awesome Generative Recommendation

[![PR Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/hyp1231/awesome-generative-recommendation/pulls)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

---

📌 Pinned

* WWW 2025 Tutorial on generative recommendation models based on **LLMs**, **semantic IDs**, and **diffusion** models. [[website](large-genrec.github.io/www2025.html)]

<center><strong>
Day 2: Tuesday, April 29, 2025, 9:00 AM - 12:30 PM (AEST), in Room C2.4
</strong></center>

---

- [Papers](#papers)
  - [Surveys](#surveys)
  - [LLM-based Generative Recommendation](#llm-based-generative-recommendation)
    - [LLM as Sequential Recommender](#llm-as-sequential-recommender)
        - [Early Efforts: Zero-shot Recommendation with LLMs](#early-efforts-zero-shot-recommendation-with-llms)
        - [Aligning LLMs for Recommendation](#aligning-llms-for-recommendation)
        - [Training Objectives & Inference](#training-objectives--inference)
    - [LLM as Conversational Recommender & Recommendation Assistant](#llm-as-conversational-recommender--recommendation-assistant)
    - [LLM as User Simulator](#llm-as-user-simulator)
  - [Semantic ID-based Generative Recommendation](#semantic-id-based-generative-recommendation)
    - [Semantic ID Construction](#semantic-id-construction)
        - [Quantization](#quantization)
        - [Hierarchical Clustering](#hierarchical-clustering)
        - [Contextual Action Tokenization](#contextual-action-tokenization)
        - [Behavior-aware Tokenization](#behavior-aware-tokenization)
        - [Language Model-based Generator](#language-model-based-generator)
    - [Architecture](#architecture)
        - [Dense & Generative Retrieval](#dense--generative-retrieval)
        - [Unified Retrieval and Ranking](#unified-retrieval-and-ranking)
    - [Aligning with LLMs](#aligning-with-llms)
  - [Diffusion Model-based Generative Recommendation](#diffusion-model-based-generative-recommendation)
    - [Diffusion-enhanced Recommendation](#diffusion-enhanced-recommendation)
    - [Diffusion as Recommender](#diffusion-as-recommender)
    - [Personalized Content Generation with Diffusion](#personalized-content-generation-with-diffusion)
- [Resources](#resources)


## Papers

### Surveys

* **A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys).** KDD 2024. [[paper](https://arxiv.org/abs/2404.00579)]

    *Yashar Deldjoo, Zhankui He, Julian McAuley, Anton Korikov, Scott Sanner, Arnau Ramisa, René Vidal, Maheswaran Sathiamoorthy, Atoosa Kasirzadeh, Silvia Milano.*

### LLM-based Generative Recommendation

#### LLM as Sequential Recommender

##### Early Efforts: Zero-shot Recommendation with LLMs

* (LLMRank) **Large language models are zero-shot rankers for recommender systems.** ECIR 2024. [[paper](https://arxiv.org/pdf/2305.08845)] [[code](https://github.com/RUCAIBox/LLMRank)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/LLMRank)

   *Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, Wayne Xin Zhao.*

* (Chat-REC) **Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System.** arXiv:2303.14524. [[paper](https://arxiv.org/pdf/2303.14524)] 

   *Yunfan Gao, Tao Sheng, Youlin Xiang, Yun Xiong, Haofen Wang, Jiawei Zhang.*

* (NIR) **Zero-Shot Next-Item Recommendation using Large Pretrained Language Models.** arXiv:2304.03153. [[paper](https://arxiv.org/pdf/2304.03153)] [[code](https://github.com/AGI-Edgerunners/LLM-Next-Item-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/AGI-Edgerunners/LLM-Next-Item-Rec)

   *Lei Wang, Ee-Peng Lim.*

* (ChatNews) **A Preliminary Study of ChatGPT on News Recommendation: Personalization, Provider Fairness, Fake News.** arXiv:2306.10702. [[paper](https://arxiv.org/pdf/2306.10702)] [[code](https://github.com/imrecommender/ChatGPT-News)] ![GitHub Repo stars](https://img.shields.io/github/stars/imrecommender/ChatGPT-News)

   *Xinyi Li, Yongfeng Zhang, Edward C. Malthouse.*

* **The Unequal Opportunities of Large Language Models: Revealing Demographic Bias through Job Recommendations.** arXiv:2308.02053. [[paper](https://arxiv.org/pdf/2308.02053)] [[code](https://github.com/Abel2Code/Unequal-Opportunities-of-LLMs)] ![GitHub Repo stars](https://img.shields.io/github/stars/Abel2Code/Unequal-Opportunities-of-LLMs)

   *Abel Salinas, Parth Vipul Shah, Yuzhong Huang, Robert McCormack, Fred Morstatter.*

* **Is ChatGPT a Good Recommender? A Preliminary Study.** CIKM 2023. [[paper](https://arxiv.org/pdf/2304.10149)] [[code](https://github.com/williamliujl/LLMRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/williamliujl/LLMRec)

   *Junling Liu, Chao Liu, Peilin Zhou, Renjie Lv, Kang Zhou, Yan Zhang.*

* **Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation.** RecSys 2023. [[paper](https://arxiv.org/pdf/2305.07609)] [[code](https://github.com/jizhi-zhang/FaiRLLM)] ![GitHub Repo stars](https://img.shields.io/github/stars/jizhi-zhang/FaiRLLM)

   *Jizhi Zhang, Keqin Bao, Yang Zhang, Wenjie Wang, Fuli Feng, Xiangnan He.*

* **Uncovering ChatGPT's Capabilities in Recommender Systems.** RecSys 2023. [[paper](https://arxiv.org/pdf/2305.02182)] [[code](https://github.com/rainym00d/LLM4RS)] ![GitHub Repo stars](https://img.shields.io/github/stars/rainym00d/LLM4RS)

   *Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, Jun Xu.*

* **Leveraging Large Language Models for Sequential Recommendation.** RecSys 2023. [[paper](https://arxiv.org/pdf/2309.09261)] [[code](https://github.com/dh-r/LLM-Sequential-Recommendation)] ![GitHub Repo stars](https://img.shields.io/github/stars/dh-r/LLM-Sequential-Recommendation)

   *Jesse Harte, Wouter Zorgdrager, Panos Louridas, Asterios Katsifodimos, Dietmar Jannach, Marios Fragkoulis.*

* (Rec-GPT4V) **Rec-GPT4V: Multimodal Recommendation with Large Vision-Language Models.** arXiv:2402.08670. [[paper](https://arxiv.org/pdf/2402.08670)]

   *Yuqing Liu, Yu Wang, Lichao Sun, Philip S. Yu.*

##### Aligning LLMs for Recommendation

* (TallRec) **TallRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation.** RecSys 2023. [[paper](https://arxiv.org/pdf/2305.00447)] [[code](https://github.com/SAI990323/TALLRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/SAI990323/TALLRec)

   *Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, Xiangnan He.*

* (GPT4Rec) **GPT4Rec: A Generative Framework for Personalized Recommendation and User Interests Interpretation.** arXiv:2304.03879. [[paper](https://arxiv.org/pdf/2304.03879)]

   *Jinming Li, Wentao Zhang, Tian Wang, Guanglei Xiong, Alan Lu, Gerard Medioni.*

* (M6-Rec) **M6-Rec: Generative Pretrained Language Models are Open-Ended Recommender Systems.** arXiv:2205.08084. [[paper](https://arxiv.org/pdf/2205.08084)]

   *Zeyu Cui, Jianxin Ma, Chang Zhou, Jingren Zhou, Hongxia Yang.*

* (BIGRec) **A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems.** arXiv:2308.08434. [[paper](https://arxiv.org/pdf/2308.08434)] [[code](https://github.com/SAI990323/BIGRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/SAI990323/BIGRec)

   *Keqin Bao, Jizhi Zhang, Wenjie Wang, Yang Zhang, Zhengyi Yang, Yancheng Luo, Chong Chen, Fuli Feng, Qi Tian.*

* (InstructRec) **Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach.** TOIS 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3708882)]

   *Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, Ji-Rong Wen.*

* (P5) **Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5).** RecSys 2022. [[paper](https://arxiv.org/pdf/2203.13366)] [[code](https://github.com/jeykigung/P5)] ![GitHub Repo stars](https://img.shields.io/github/stars/jeykigung/P5)

   *Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, Yongfeng Zhang.*

* (VIP5) **Towards Multimodal Foundation Models for Recommendation.** arXiv:2305.14302. [[paper](https://arxiv.org/pdf/2305.14302)] [[code](https://github.com/jeykigung/VIP5)] ![GitHub Repo stars](https://img.shields.io/github/stars/jeykigung/VIP5)

   *Shijie Geng, Juntao Tan, Shuchang Liu, Zuohui Fu, Yongfeng Zhang.*

* (GenRec) **Generative Recommendation: Towards Next-generation Recommender Paradigm.** arXiv:2304.03516. [[paper](https://arxiv.org/pdf/2304.03516)] [[code](https://github.com/Linxyhaha/GeneRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/GeneRec)

    *Wenjie Wang, Xinyu Lin, Fuli Feng, Xiangnan He, Tat-Seng Chua.*

* (P5-ID)**How to Index Item IDs for Recommendation Foundation Models.** SIGIR-AP 2023. [[paper](https://dl.acm.org/doi/pdf/10.1145/3624918.3625339)] [[code](https://github.com/Wenyueh/LLM-RecSys-ID)] ![GitHub Repo stars](https://img.shields.io/github/stars/Wenyueh/LLM-RecSys-ID)

    *Wenyue Hua, Shuyuan Xu, Yingqiang Ge, Yongfeng Zhang.*

* (HKFR) **Heterogeneous Knowledge Fusion: A Novel Approach for Personalized Recommendation via LLM.** RecSys 2023. [[paper](https://arxiv.org/pdf/2308.03333)]
   *Bin Yin, Junjie Xie, Yu Qin, Zixiang Ding, Zhichao Feng, Xiang Li, Wei Lin.*

* (LlamaRec) **LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking.** PGAI@CIKM 2023. [[paper](https://arxiv.org/pdf/2311.02089)] [[code](https://github.com/Yueeeeeeee/LlamaRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Yueeeeeeee/LlamaRec)
   *Zhenrui Yue, Sara Rabhi, Gabriel de Souza Pereira Moreira, Dong Wang, Even Oldridge.*

* (ReLLa) **ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation.** arXiv:2308.11131. [[paper](https://arxiv.org/pdf/2308.11131)] [[code](https://github.com/LaVieEnRose365/ReLLa)] ![GitHub Repo stars](https://img.shields.io/github/stars/LaVieEnRose365/ReLLa)

    *Jianghao Lin, Rong Shan, Chenxu Zhu, Kounianhua Du, Bo Chen, Shigang Quan, Ruiming Tang, Yong Yu, and Weinan Zhang.*

* (DEALRec) **Data-efficient Fine-tuning for LLM-based Recommendation.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2401.17197)] [[code](https://github.com/Linxyhaha/DEALRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/DEALRec)

    *Xinyu Lin, Wenjie Wang, Yongqi Li, Shuo Yang, Fuli Feng, Yinwei Wei, Tat-Seng Chua.*

* (CLLM4Rec) **Collaborative Large Language Model for Recommender Systems.** WWW 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3589334.3645347)] [[code](https://github.com/yaochenzhu/llm4rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/yaochenzhu/llm4rec)

    *Yaochen Zhu, Liang Wu, Qi Guo, Liangjie Hong, Jundong Li.*

* (RecInterpreter) **Large Language Model Can Interpret Latent Space of Sequential Recommender.** arXiv:2310.20487. [[paper](https://arxiv.org/pdf/2310.20487)] [[code](https://github.com/YangZhengyi98/RecInterpreter)] ![GitHub Repo stars](https://img.shields.io/github/stars/YangZhengyi98/RecInterpreter)

    *Zhengyi Yang, Jiancan Wu, Yanchen Luo, Jizhi Zhang, Yancheng Yuan, An Zhang, Xiang Wang, Xiangnan He.*

* (TransRec) **Bridging Items and Language: A Transition Paradigm for Large Language Model-Based Recommendation.** KDD 2024. [[paper](https://arxiv.org/pdf/2310.06491)] [[code](https://github.com/Linxyhaha/TransRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/TransRec)

   *Xinyu Lin, Wenjie Wang, Yongqi Li, Fuli Feng, See-Kiong Ng, Tat-Seng Chua.*

* (RecExplainer) **RecExplainer: Aligning Large Language Models for Explaining Recommendation Models.** KDD 2024. [[paper](https://arxiv.org/pdf/2311.10947)] [[code](https://github.com/microsoft/RecAI)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/RecAI)

   *Yuxuan Lei, Jianxun Lian, Jing Yao, Xu Huang, Defu Lian, Xing Xie.*

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

* (ToolRec) **Let Me Do It For You: Towards LLM Empowered Recommendation via Tool Learning.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2405.15114)]

   *Yuyue Zhao, Jiancan Wu, Xiang Wang, Wei Tang, Dingxian Wang, Maarten de Rijke.*

* (LLaRA) **LLaRA: Large Language-Recommendation Assistant.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2312.02445)] [[code](https://github.com/ljy0ustc/LLaRA)] ![GitHub Repo stars](https://img.shields.io/github/stars/ljy0ustc/LLaRA)

   *Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, Xiang Wang, Xiangnan He.*

* (I-LLMRec) **Image is All You Need: Towards Efficient and Effective Large Language Model-Based Recommender Systems.** arXiv:2503.06238. [[paper](https://arxiv.org/pdf/2503.06238)] [[code](https://github.com/rlqja1107/torch-I-LLMRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/rlqja1107/torch-I-LLMRec)

   *Kibum Kim, Sein Kim, Hongseok Kang, Jiwan Kim, Heewoong Noh, Yeonjun In, Kanghoon Yoon, Jinoh Oh, Chanyoung Park.*

* (RALLRec+) **RALLRec+: Retrieval Augmented Large Language Model Recommendation with Reasoning.** arXiv:2503.20430. [[paper](https://arxiv.org/pdf/2503.20430)] [[code](https://github.com/sichunluo/RALLRec_plus)] ![GitHub Repo stars](https://img.shields.io/github/stars/sichunluo/RALLRec_plus)

   *Sichun Luo, Jian Xu, Xiaojie Zhang, Linrong Wang, Sicong Liu, Hanxu Hou, Linqi Song.*

##### Training Objectives & Inference

* (S-DPO) **On Softmax Direct Preference Optimization for Recommendation.** NeurIPS 2024. [[paper](https://arxiv.org/pdf/2406.09215)] [[code](https://github.com/chenyuxin1999/S-DPO)] ![GitHub Repo stars](https://img.shields.io/github/stars/chenyuxin1999/S-DPO)

   *Yuxin Chen, Junfei Tan, An Zhang, Zhengyi Yang, Leheng Sheng, Enzhi Zhang, Xiang Wang, Tat-Seng Chua.*

* (D3) **Decoding Matters: Addressing Amplification Bias and Homogeneity Issue for LLM-based Recommendation.** EMNLP 2024. [[paper](https://arxiv.org/pdf/2406.14900)] [[code](https://github.com/SAI990323/DecodingMatters)] ![GitHub Repo stars](https://img.shields.io/github/stars/SAI990323/DecodingMatters)

   *Keqin Bao, Jizhi Zhang, Yang Zhang, Xinyue Huo, Chong Chen, Fuli Feng.*

* (SLMREC) **SLMREC: Empowering Small Language Models for Sequential Recommendation.** arXiv:2405.17890. [[paper](https://arxiv.org/pdf/2405.17890)] [[code](https://github.com/WujiangXu/SLMRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/WujiangXu/SLMRec)

   *Wujiang Xu, Qitian Wu, Zujie Liang, Jiaojiao Han, Xuying Ning, Yunxiao Shi, Wenfang Lin, Yongfeng Zhang.*


#### LLM as Conversational Recommender & Recommendation Assistant

* (LLM-REDIAL) **LLM-REDIAL: A Large-Scale Dataset for Conversational Recommender Systems Created from User Behaviors with LLMs.** ACL Findings 2024. [[paper](https://aclanthology.org/2024.findings-acl.529.pdf)] [[code](https://github.com/LitGreenhand/LLM-Redial)] ![GitHub Repo stars](https://img.shields.io/github/stars/LitGreenhand/LLM-Redial)

   *Tingting Liang, Chenxin Jin, Lingzhi Wang, Wenqi Fan, Congying Xia, Kai Chen, Yuyu Yin.*

* (iEvaLM) **Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models.** EMNLP 2023. [[paper](https://arxiv.org/pdf/2305.13112)] [[code](https://github.com/RUCAIBox/iEvaLM-CRS)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/iEvaLM-CRS)

   *Xiaolei Wang, Xinyu Tang, Wayne Xin Zhao, Jingyuan Wang, Ji-Rong Wen.*

*  **How Reliable is Your Simulator? Analysis on the Limitations of Current LLM-based User Simulators for Conversational Recommendation.** WWW 2024. [[paper](https://arxiv.org/pdf/2403.16416)] [[code](https://github.com/RUCAIBox/iEvaLM-CRS)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/iEvaLM-CRS)

   *Lixi Zhu, Xiaowen Huang, Jitao Sang.*

* **Large Language Models as Zero-Shot Conversational Recommenders.** CIKM 2023. [[paper](https://dl.acm.org/doi/pdf/10.1145/3583780.3614949)] [[code](https://github.com/AaronHeee/LLMs-as-Zero-Shot-Conversational-RecSys)] ![GitHub Repo stars](https://img.shields.io/github/stars/AaronHeee/LLMs-as-Zero-Shot-Conversational-RecSys)

   *Zhankui He, Zhouhang Xie, Rahul Jha, Harald Steck, Dawen Liang, Yesu Feng, Bodhisattwa Prasad Majumder, Nathan Kallus, Julian McAuley.*

* (InteRecAgent) **Recommender AI Agent: Integrating Large Language Models for Interactive Recommendations.** arXiv:2308.16505. [[paper](https://arxiv.org/pdf/2308.16505)] [[code](https://github.com/microsoft/RecAI)] ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/RecAI)

   *Xu Huang, Jianxun Lian, Yuxuan Lei, Jing Yao, Defu Lian, Xing Xie.*

* (RecMind) **RecMind: Large Language Model Powered Agent For Recommendation.** NACCL 2024 (Findings). [[paper](https://arxiv.org/pdf/2308.14296)]

   *Yancheng Wang, Ziyan Jiang, Zheng Chen, Fan Yang, Yingxue Zhou, Eunah Cho, Xing Fan, Xiaojiang Huang, Yanbin Lu, Yingzhen Yang.*

* (RAH) **RAH! RecSys–Assistant–Human: A Human-Centered Recommendation Framework With LLM Agents.** TOCS 2024. [[paper](https://arxiv.org/pdf/2308.09904)]

   *Yubo Shu, Haonan Zhang, Hansu Gu, Peng Zhang, Tun Lu, Dongsheng Li, Ning Gu.*

* (MACRec) **MACRec: A Multi-Agent Collaboration Framework for Recommendation.** SIGIR 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3626772.3657669)] [[code](https://github.com/wzf2000/MACRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/wzf2000/MACRec)

   *Zhefan Wang, Yuanqing Yu, Wendi Zheng, Weizhi Ma, Min Zhang.*


#### LLM as User Simulator

* (RecAgent) **User Behavior Simulation with Large Language Model-based Agents for Recommender Systems.** TOIS 2024. [[paper](https://dl.acm.org/doi/pdf/10.1145/3708985)] [[code](https://github.com/RUC-GSAI/YuLan-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUC-GSAI/YuLan-Rec)

   *Lei Wang, Jingsen Zhang, Hao Yang, Zhi-Yuan Chen, Jiakai Tang, Zeyu Zhang, Xu Chen, Yankai Lin, Hao Sun, Ruihua Song, Wayne Xin Zhao, Jun Xu, Zhicheng Dou, Jun Wang, Ji-Rong Wen.*

* (AgentCF) **AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems.** WWW 2024. [[paper](https://arxiv.org/pdf/2310.09233)]

   *Junjie Zhang, Yupeng Hou, Ruobing Xie, Wenqi Sun, Julian McAuley, Wayne Xin Zhao, Leyu Lin, Ji-Rong Wen.*

* (Agent4Rec) **On Generative Agents in Recommendation.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2310.10108)] [[code](https://github.com/LehengTHU/Agent4Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/LehengTHU/Agent4Rec)

   *An Zhang, Yuxin Chen, Leheng Sheng, Xiang Wang, Tat-Seng Chua.*

* **LLM-Powered User Simulator for Recommender System.** arXiv:2412.16984. [[paper](https://arxiv.org/pdf/2412.16984)]

   *Zijian Zhang, Shuchang Liu, Ziru Liu, Rui Zhong, Qingpeng Cai, Xiangyu Zhao, Chunxu Zhang, Qidong Liu, Peng Jiang.*

* **Enhancing Cross-Domain Recommendations with Memory-Optimized LLM-Based User Agents.** arXiv:2502.13843. [[paper](https://arxiv.org/pdf/2502.13843)]

   *Jiahao Liu, Shengkang Gu, Dongsheng Li, Guangping Zhang, Mingzhe Han, Hansu Gu, Peng Zhang, Tun Lu, Li Shang, Ning Gu.*

* **FLOW: A Feedback LOop FrameWork for Simultaneously Enhancing Recommendation and User Agents.** arXiv:2410.20027. [[paper](https://arxiv.org/pdf/2410.20027)]

   *Shihao Cai, Jizhi Zhang, Keqin Bao, Chongming Gao, Fuli Feng.*

* **A LLM-based Controllable, Scalable, Human-Involved User Simulator Framework for Conversational Recommender Systems.** arXiv:2405.08035. [[paper](https://arxiv.org/pdf/2405.08035)] [[code](https://github.com/zlxxlz1026/CSHI)] ![GitHub Repo stars](https://img.shields.io/github/stars/zlxxlz1026/CSHI)

   *Lixi Zhu, Xiaowen Huang, Jitao Sang.*


### Semantic ID-based Generative Recommendation

#### Semantic ID Construction

##### Quantization

* (VQ-Rec) **Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders.** WWW 2023. [[paper](https://arxiv.org/abs/2210.12316)] [[code](https://github.com/RUCAIBox/VQ-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/VQ-Rec)

   *Yupeng Hou, Zhankui He, Julian McAuley, Wayne Xin Zhao.*

* (TIGER) **Recommender Systems with Generative Retrieval.** NeurIPS 2023. [[paper](https://arxiv.org/abs/2305.05065)]

   *Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan H. Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Q. Tran, Jonah Samost, Maciej Kula, Ed H. Chi, Maheswaran Sathiamoorthy.*

* **Generative Sequential Recommendation with GPTRec.** Gen-IR @ SIGIR 2023 workshop. [[paper](https://arxiv.org/abs/2306.11114)]

   *Aleksandr V. Petrov, Craig Macdonald.*

* (ColaRec) **Content-Based Collaborative Generation for Recommender Systems.** CIKM 2024. [[paper](https://arxiv.org/abs/2403.18480)] [[code](https://github.com/Junewang0614/ColaRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Junewang0614/ColaRec)

    *Yidan Wang, Zhaochun Ren, Weiwei Sun, Jiyuan Yang, Zhixiang Liang, Xin Chen, Ruobing Xie, Su Yan, Xu Zhang, Pengjie Ren, Zhumin Chen, Xin Xin.*

* **CoST: Contrastive Quantization based Semantic Tokenization for Generative Recommendation.** RecSys 2024. [[paper](https://arxiv.org/abs/2404.14774)]

    *Jieming Zhu, Mengqun Jin, Qijiong Liu, Zexuan Qiu, Zhenhua Dong, Xiu Li.*

* **MMGRec: Multimodal Generative Recommendation with Transformer Model.** arXiv:2404.16555. [[paper](https://arxiv.org/abs/2404.16555)]

    *Han Liu, Yinwei Wei, Xuemeng Song, Weili Guan, Yuan-Fang Li, Liqiang Nie.*

* (LETTER) **Learnable Item Tokenization for Generative Recommendation.** CIKM 2024. [[paper](https://arxiv.org/abs/2405.07314)] [[code](https://github.com/HonghuiBao2000/LETTER)] ![GitHub Repo stars](https://img.shields.io/github/stars/HonghuiBao2000/LETTER)

    *Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, See-Kiong Ng, Tat-Seng Chua.*

* (ETEGRec) **End-to-End Learnable Item Tokenization for Generative Recommendation.** arXiv:2409.05546. [[paper](https://arxiv.org/abs/2409.05546)]

    *Enze Liu, Bowen Zheng, Cheng Ling, Lantao Hu, Han Li, Wayne Xin Zhao.*

* (MoC) **Towards Scalable Semantic Representation for Recommendation.** arXiv:2410.09560. [[paper](https://arxiv.org/abs/2410.09560)]

    *Taolin Zhang, Junwei Pan, Jinpeng Wang, Yaohua Zha, Tao Dai, Bin Chen, Ruisheng Luo, Xiaoxiang Deng, Yuan Wang, Ming Yue, Jie Jiang, Shu-Tao Xia.*

##### Hierarchical Clustering

* (DSI) **Transformer Memory as a Differentiable Search Index.** NeurIPS 2022. [[paper](https://arxiv.org/abs/2202.06991)]

   *Yi Tay, Vinh Q. Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, Tal Schuster, William W. Cohen, Donald Metzler.*

* (NCI) **A Neural Corpus Indexer for Document Retrieval.** NeurIPS 2022. [[paper](https://arxiv.org/abs/2206.02743)] [[code](https://github.com/solidsea98/Neural-Corpus-Indexer-NCI)] ![GitHub Repo stars](https://img.shields.io/github/stars/solidsea98/Neural-Corpus-Indexer-NCI)

   *Yujing Wang, Yingyan Hou, Haonan Wang, Ziming Miao, Shibin Wu, Hao Sun, Qi Chen, Yuqing Xia, Chengmin Chi, Guoshuai Zhao, Zheng Liu, Xing Xie, Hao Allen Sun, Weiwei Deng, Qi Zhang, Mao Yang.*

* **How to Index Item IDs for Recommendation Foundation Models.** SIGIR-AP 2023. [[paper](https://arxiv.org/abs/2305.06569)] [[code](https://github.com/Wenyueh/LLM-RecSys-ID)] ![GitHub Repo stars](https://img.shields.io/github/stars/Wenyueh/LLM-RecSys-ID)

   *Wenyue Hua, Shuyuan Xu, Yingqiang Ge, Yongfeng Zhang.*

* (SEATER) **Generative Retrieval with Semantic Tree-Structured Item Identifiers via Contrastive Learning.** SIGIR-AP 2024. [[paper](https://arxiv.org/abs/2309.13375)] [[code](https://github.com/ethan00si/seater_generative_retrieval)] ![GitHub Repo stars](https://img.shields.io/github/stars/ethan00si/seater_generative_retrieval)

    *Zihua Si, Zhongxiang Sun, Jiale Chen, Guozhang Chen, Xiaoxue Zang, Kai Zheng, Yang Song, Xiao Zhang, Jun Xu, Kun Gai.*

##### Contextual Action Tokenization

* **ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation.** arXiv:2502.13581. [[paper](https://arxiv.org/abs/2502.13581)]

    *Yupeng Hou, Jianmo Ni, Zhankui He, Noveen Sachdeva, Wang-Cheng Kang, Ed H. Chi, Julian McAuley, Derek Zhiyuan Cheng.*

##### Behavior-aware Tokenization

* **EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration.** KDD 2024. [[paper](https://arxiv.org/abs/2406.14017)] [[code](https://github.com/yewzz/EAGER)] ![GitHub Repo stars](https://img.shields.io/github/stars/yewzz/EAGER)

    *Ye Wang, Jiahao Xun, Minjie Hong, Jieming Zhu, Tao Jin, Wang Lin, Haoyuan Li, Linjun Li, Yan Xia, Zhou Zhao, Zhenhua Dong.*

* **SC-Rec: Enhancing Generative Retrieval with Self-Consistent Reranking for Sequential Recommendation.** arXiv:2408.08686. [[paper](https://arxiv.org/abs/2408.08686)]

    *Tongyoung Kim, Soojin Yoon, Seongku Kang, Jinyoung Yeo, Dongha Lee.*

* (MBGen) **Multi-Behavior Generative Recommendation.** CIKM 2024. [[paper](https://arxiv.org/abs/2405.16871)] [[code](https://github.com/anananan116/MBGen)] ![GitHub Repo stars](https://img.shields.io/github/stars/anananan116/MBGen)

    *Zihan Liu, Yupeng Hou, Julian McAuley.*

* (PRORec) **Progressive Collaborative and Semantic Knowledge Fusion for Generative Recommendation.** arXiv:2502.06269. [[paper](https://arxiv.org/abs/2502.06269)]

    *Longtao Xiao, Haozhao Wang, Cheng Wang, Linfei Ji, Yifan Wang, Jieming Zhu, Zhenhua Dong, Rui Zhang, Ruixuan Li.*

##### Language Model-based Generator

* (LMIndexer) **Language Models As Semantic Indexers.** ICDE 2024. [[paper](https://arxiv.org/abs/2310.07815)] [[code](https://github.com/PeterGriffinJin/LMIndexer)] ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/LMIndexer)

    *Bowen Jin, Hansi Zeng, Guoyin Wang, Xiusi Chen, Tianxin Wei, Ruirui Li, Zhengyang Wang, Zheng Li, Yang Li, Hanqing Lu, Suhang Wang, Jiawei Han, Xianfeng Tang.*

* **IDGenRec: LLM-RecSys Alignment with Textual ID Learning.** SIGIR 2024. [[paper](https://arxiv.org/abs/2403.19021)] [[code](https://github.com/agiresearch/IDGenRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/agiresearch/IDGenRec)

    *Juntao Tan, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Zelong Li, Yongfeng Zhang.*

#### Architecture

##### Dense & Generative Retrieval

* (SpecGR) **Inductive Generative Recommendation via Retrieval-based Speculation.** arXiv:2410.02939. [[paper](https://arxiv.org/abs/2410.02939)] [[code](https://github.com/Jamesding000/SpecGR)] ![GitHub Repo stars](https://img.shields.io/github/stars/Jamesding000/SpecGR)

    *Yijie Ding, Yupeng Hou, Jiacheng Li, Julian McAuley.*

* (LIGER) **Unifying Generative and Dense Retrieval for Sequential Recommendation.** arXiv:2411.18814. [[paper](https://arxiv.org/abs/2411.18814)]

    *Liu Yang, Fabian Paischer, Kaveh Hassani, Jiacheng Li, Shuai Shao, Zhang Gabriel Li, Yun He, Xue Feng, Nima Noorshams, Sem Park, Bo Long, Robert D Nowak, Xiaoli Gao, Hamid Eghbalzadeh.*

##### Unified Retrieval and Ranking

* (HSTU) **Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations.** ICML 2024. [[paper](https://arxiv.org/abs/2402.17152)] [[code](https://github.com/facebookresearch/generative-recommenders)] ![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/generative-recommenders)

    *Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Michael He, Yinghai Lu, Yu Shi.*

* **OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment.** arXiv:2502.18965. [[paper](https://arxiv.org/abs/2502.18965)]

    *Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, Guorui Zhou.*

#### Aligning with LLMs

* **Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5).** RecSys 2022. [[paper](https://arxiv.org/abs/2203.13366)] [[code](https://github.com/jeykigung/P5)] ![GitHub Repo stars](https://img.shields.io/github/stars/jeykigung/P5)

   *Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, Yongfeng Zhang.*

* (LC-Rec) **Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation.** ICDE 2024. [[paper](https://arxiv.org/abs/2311.09049)] [[code](https://github.com/RUCAIBox/LC-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/LC-Rec)

    *Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, Ji-Rong Wen.*

* (AtSpeed) **Efficient Inference for Large Language Model-based Generative Recommendation.** ICLR 2025. [[paper](https://arxiv.org/abs/2410.05165)] [[code](https://github.com/Linxyhaha/AtSpeed)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/AtSpeed)

    *Xinyu Lin, Chaoqun Yang, Wenjie Wang, Yongqi Li, Cunxiao Du, Fuli Feng, See-Kiong Ng, Tat-Seng Chua.*

* **Semantic Convergence: Harmonizing Recommender Systems via Two-Stage Alignment and Behavioral Semantic Tokenization.** AAAI 2025. [[paper](https://arxiv.org/abs/2412.13771)]

    *Guanghan Li, Xun Zhang, Yufei Zhang, Yifan Yin, Guojun Yin, Wei Lin.*

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

* **Diffkg: Knowledge Graph Diffusion Model for Recommendation.** WSDM 2024. [[paper](https://arxiv.org/abs/2312.16890)]

    *Yangqin Jiang, Yuhao Yang, Lianghao Xia, Chao Huang.*



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


* **Recommendation via Collaborative Diffusion Generative Model.** KSEM 2022. [[paper]([https://link.springer.com/chapter/10.1007/978-3-031-10989-8_49](https://link.springer.com/chapter/10.1007/978-3-031-10989-8_47))]

    *Joojo Walker, Ting Zhong, Fengli Zhang, Qiang Gao, Fan Zhou.*

* **G-Diff: A Graph-Based Decoding Network for Diffusion Recommender Model.** IEEE TNNLS 2024. [[paper](https://ieeexplore.ieee.org/document/10750895)]

    *Ruixin Chen, Jianping Fan, Meiqin Wu, Rui Cheng, Jiawen Song.*

* **Collaborative Filtering Based on Diffusion Models: Unveiling the Potential of High-Order Connectivity.** SIGIR 2024. [[paper](https://arxiv.org/abs/2404.14240)] [[code](https://github.com/jackfrost168/CF_Diff)]

    *Yu Hou, Jin-Duk Park, Won-Yong Shin.*

* **Denoising Diffusion Recommender Model.** SIGIR 2024. [[paper](https://arxiv.org/abs/2401.06982)] [[code](https://github.com/Polaris-JZ/DDRM)]

    *Jujia Zhao, Wang Wenjie, Yiyan Xu, Teng Sun, Fuli Feng, Tat-Seng Chua.*

* **DGRM: Diffusion-GAN Recommendation Model to Alleviate the Mode Collapse Problem in Sparse Environments.** Pattern Recognition 2024. [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320324004436)]

    *Deng Jiangzhou, Wang Songli, Ye Jianmei, Ji Lianghao, Wang Yong.*

* **Stochastic Sampling for Contrastive Views and Hard Negative Samples in Graph-based Collaborative Filtering.** WSDM 2025. [[paper](https://arxiv.org/abs/2405.00287)] [[code](https://github.com/jeongwhanchoi/SCONE)]

    *Chaejeong Lee, Jeongwhan Choi, Hyowon Wi, Sung-Bae Cho, Noseong Park.*

* **RecDiff: Diffusion Model for Social Recommendation.** CIKM 2024. [[paper](https://arxiv.org/abs/2406.01629)] [[code](https://github.com/HKUDS/RecDiff)]

    *Zongwei Li, Lianghao Xia, Chao Huang.*

* **Blurring-Sharpening Process Models for Collaborative Filtering.** SIGIR 2023. [[paper](https://arxiv.org/abs/2211.09324)]

    *Jeongwhan Choi, Seoyoung Hong, Noseong Park, Sung-Bae Cho.*
  
* **Graph Signal Diffusion Model for Collaborative Filtering.** SIGIR 2024. [[paper](https://arxiv.org/abs/2311.08744)]

    *Yunqin Zhu, Chao Wang, Qi Zhang, Hui Xiong.*

* **Diffurec: A Diffusion Model for Sequential Recommendation.** TOIS 2023. [[paper](https://arxiv.org/abs/2304.00686)]

    *Zihao Li, Aixin Sun, Chenliang Li.*

* **A Diffusion Model for POI Recommendation.** TOIS 2023. [[paper](https://arxiv.org/abs/2304.07041)]

    *Yifang Qin, Hongjun Wu, Wei Ju, Xiao Luo, Ming Zhang.*

* **Towards Personalized Sequential Recommendation via Guided Diffusion.** ICIC 2024. [[paper](https://dl.acm.org/doi/10.1007/978-981-97-5618-6_1)]

    *Weidong Wang, Yan Tang, Kun Tian.*

* **Diffusion Recommendation with Implicit Sequence Influence.** WebConf 2024. [[paper](https://dl.acm.org/doi/10.1145/3589335.3651951)]

    *Yong Niu, Xing Xing, Zhichun Jia, Ruidi Liu, Mindong Xin, Jianfu Cui.*

* **Uncertainty-aware Guided Diffusion for Missing Data in Sequential Recommendation.** SIGIR 2025. [[paper](https://openreview.net/forum?id=w2HL7yuWE2)]

    *Wenyu Mao, Zhengyi Yang, Jiancan Wu, Haozhe Liu, Yancheng Yuan, Xiang Wang, Xiangnan He.*

* **Generate and Instantiate What You Prefer: Text-Guided Diffusion for Sequential Recommendation.** arXiv 2024. [[paper](https://arxiv.org/abs/2410.13428)]

    *Guoqing Hu, Zhengyi Yang, Zhibo Cai, An Zhang, Xiang Wang.*


#### Personalized Content Generation with Diffusion

* **DreamVTON: Customizing 3D Virtual Try-on with Personalized Diffusion Models.** MM2024. [[paper](https://arxiv.org/abs/2407.16511)]

    *Zhenyu Xie, Haoye Dong, Yufei Gao, Zehua Ma, Xiaodan Liang.*

* **Instant 3D Human Avatar Generation using Image Diffusion Models.** ECCV 2024. [[paper](https://arxiv.org/abs/2406.07516)] [[code](https://github.com/OPPO-Mente-Lab/Subject-Diffusion)]

    *Nikos Kolotouros, Thiemo Alldieck, Enric Corona, Eduard Gabriel Bazavan, Cristian Sminchisescu.*

* **Subject-Diffusion:Open Domain Personalized Text-to-Image Generation without Test-time Fine-tuning.** SIGGRAPH 2024. [[paper](https://arxiv.org/abs/2307.11410)] [[code](https://github.com/OPPO-Mente-Lab/Subject-Diffusion)]

    *Jian Ma, Junhao Liang, Chen Chen, Haonan Lu.*

* **Diffusion Models for Generative Outfit Recommendation.** SIGIR 2024. [[paper](https://arxiv.org/abs/2402.17279)]

    *Yiyan Xu, Wenjie Wang, Fuli Feng, Yunshan Ma, Jizhi Zhang, Xiangnan He.*

## Resources

* [Awesome-Generative-RecSys](https://github.com/jihoo-kim/Awesome-Generative-RecSys) - A repo featuring papers on generative recommender systems, though not actively maintained.

## Join Our Community

We invite you to join our **Discord** community to connect with other researchers and practitioners interested in **generative recommendation models**. Share your ideas, ask questions, and stay updated on the latest developments in the field. [[invitation link](https://discord.gg/8vCbzVKr)]
