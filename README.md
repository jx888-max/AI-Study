# AI-Study
Sharing a suggested AI learning roadmap, for reference only.  Additions and suggestions are welcome.

第一节：工程

1. FastAPI

https://medium.com/@mayurakshasikdar/fastapi-genai-in-2025-building-scalable-and-intelligent-apps-with-speed-4c2b980b2c40

2. LLM部署，编排Pipeline

https://northflank.com/blog/llm-deployment-pipeline

https://blog.logrocket.com/modern-ai-stack-2025/

https://llm-d.ai/docs/architecture

3. vLLM，SGLang，TensorRT-LLM

https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html

https://docs.sglang.ai/

https://github.com/NVIDIA/TensorRT-LLM

4. 最佳实践思路参考

https://techinfotech.tech.blog/2025/06/09/best-practices-to-build-llm-tools-in-2025/

5. 数据库

https://latenode.com/blog/ai-frameworks-technical-infrastructure/vector-databases-embeddings/best-vector-databases-for-rag-complete-2025-comparison-guide

6. LLM服务

https://www.clarifai.com/blog/llm-inference-optimization/

https://medium.com/@karuturividyasagar/the-big-llm-architecture-comparison-2025-a-beginner-friendly-guide-to-the-latest-open-weight-e731318af967

https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html

7. 上下文工程

压缩

https://python.langchain.ac.cn/docs/how_to/contextual_compression/

分层行动

https://github.com/FoundationAgents/OpenManus

8. 知识图谱

(暂无链接)

第二节: 大模型基础

1. Prompt（思维链，few shot，置信度等）

https://docs.lovable.dev/prompting/prompting-one

2. RAG（查询重写/扩展，混合搜索，重排压缩）

https://datawhalechina.github.io/all-in-rag/#/chapter1/01_RAG_intro

https://dev523.medium.com/the-evolution-of-rag-how-retrieval-augmented-generation-is-transforming-enterprise-ai-in-2025-a0265bc1c297

假设性文档嵌入 (HyDE)

说明:

生成假设性文档: 利用一个LLM，根据用户查询生成一个理想化的、详细的回答文档。这个文档是“假设性”的，可能包含事实错误。

嵌入假设性文档: 将这个生成的长篇幅、富含上下文的假设性文档进行向量化嵌入。

执行检索: 使用这个假设性文档的向量在数据库中进行相似度搜索。

链接:

https://cgorale111.medium.com/hyde-hypothetical-document-embeddings-3071840e364c

https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde

https://www.falkordb.com/blog/advanced-rag/

基于LLM的查询扩展 (LLM-QE)

说明:

核心创新在于设计了一套精巧的奖励机制来优化（或对齐）用于查询扩展的LLM。

该方法采用DPO算法，通过两种奖励模型来训练扩展模型：

基于排序的奖励: 该模型评估LLM生成的扩展文档与“标准答案”文档在检索器（retriever）眼中的相似度。它直接对齐了扩展模型与检索模型的偏好，鼓励生成那些能够提升检索排名的内容。

基于答案的奖励: 该模型首先让LLM根据查询和标准答案文档生成一个“理想答案”，然后评估扩展文档与这个理想答案的语义相关性。这个机制有效地抑制了LLM为了单纯追求排序奖励而生成冗长、无关内容的倾向，使得扩展结果更精确、更聚焦于回答问题本身 。

链接:

https://arxiv.org/html/2502.17057v1

退步提示

https://www.designveloper.com/blog/advanced-rag/

https://blog.devops.dev/step-back-prompting-smarter-query-rewriting-for-higher-accuracy-rag-0eb95a9cc032

SimGRAG

https://aclanthology.org/2025.findings-acl.163/

查询分解

https://arxiv.org/html/2506.21384v1

https://arxiv.org/html/2507.18910v1

交叉编码器模型

https://arxiv.org/html/2508.08742v1

动态段落选择器

https://arxiv.org/html/2508.09497v1

上下文感知结构化与压缩

https://arxiv.org/html/2508.19357v1

CRAG (Corrective-Retrieval-Augmented Generation)

说明: 决策节点：一个轻量级的检索评估器

链接:

https://arxiv.org/abs/2401.15884

https://medium.com/@sulbha.jindal/corrective-retrieval-augmented-generation-crag-paper-review-2bf9fe0f3b31

Self-RAG

https://selfrag.github.io/

https://www.datacamp.com/tutorial/self-rag

Adaptive RAG

https://arxiv.org/abs/2403.14403

Graph RAG

https://msdocs.cn/graphrag/

自适应检索

https://arxiv.org/abs/2504.05312

混合架构

https://arxiv.org/html/2509.12765v1

Reasoning RAG

https://arxiv.org/html/2506.10408v1

3. Multi-Agent
适合入门的agent教程
https://github.com/datawhalechina/hello-agents

https://www.anthropic.com/engineering/multi-agent-research-system

https://collabnix.com/multi-agent-multi-llm-systems-the-future-of-ai-architecture-complete-guide-2025/

https://github.com/microsoft/autogen

https://github.com/context-machine-lab/ContextAgent

4. 标准化协议

MCP

https://www.descope.com/learn/post/mcp

https://docs.mcpcn.org/introduction

https://mcpmarket.cn/

A2A

https://a2a-protocol.org/latest/#how-does-a2a-work-with-mcp

https://codelabs.developers.google.com/intro-a2a-purchasing-concierge?hl=zh-cn#0

5. LangGraph

https://docs.langchain.com/oss/python/langgraph/overview

6，llm入门
https://github.com/datawhalechina/happy-llm

第三节：基础算法

1. 微调 (Fine-tuning)

https://medium.com/@pradeepdas/the-fine-tuning-landscape-in-2025-a-comprehensive-analysis-d650d24bed97

https://www.ibm.com/think/topics/parameter-efficient-fine-tuning

https://www.geeksforgeeks.org/deep-learning/what-is-fine-tuning/

https://github.com/hiyouga/LLaMA-Factory

https://github.com/huggingface/peft

https://github.com/dvgodoy/FineTuningLLMs

https://github.com/unslothai/notebooks

https://github.com/unslothai/unsloth

2. 强化学习 (RL)

https://www.gocodeo.com/post/top-5-reinforcement-learning-courses-in-2025-learn-from-basics-to-advanced-practice

https://github.com/Ronchy2000/Multi-agent-RL

https://www.marl-book.com/download/marl-book.pdf

https://apxml.com/zh/courses/multi-agent-llm-systems-design-implementation/chapter-1-core-principles-multi-agent-llm/architectural-frameworks-llm-mas

https://arxiv.org/html/2508.04652v1

https://arxiv.org/html/2510.04935v1

https://www.researchgate.net/publication/388920733_A_Survey_on_Explainable_Deep_Reinforcement_Learning

https://github.com/uncbiag/Awesome-Foundation-Models

https://github.com/Zehong-Wang/Awesome-Foundation-Models-on-Graphs

https://www.findingtheta.com/blog/aligning-and-augmenting-intelligence-a-technical-survey-of-reinforcement-learning-in-large-language-models

3. 预训练

https://github.com/RUCAIBox/awesome-llm-pretraining

4. 介绍了Instagram

https://www.analyticsvidhya.com/blog/2025/10/how-instagram-leverages-ai-for-content-moderation/

5. NLP

https://www.geeksforgeeks.org/machine-learning/best-python-libraries-for-machine-learning/

https://www.geeksforgeeks.org/nlp/history-and-evolution-of-nlp/

https://github.com/huggingface/transformers

6. 视觉

https://arxiv.org/html/2503.04641v1

7，DeepResearch
https://huggingface.co/FractalAIResearch/Fathom-Search-4B

第四节：LLM案例，新闻

https://www.softwebsolutions.com/resources/llm-use-cases/

https://www.rapidops.com/blog/top-groundbreaking-llm-use-cases/

https://www.rundown.ai/

https://techcrunch.com/category/artificial-intelligence/
