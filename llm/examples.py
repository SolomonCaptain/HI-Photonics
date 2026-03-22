"""
LLM 助手使用示例

演示如何使用 LLM 助手进行逆向设计辅助。
"""

import asyncio
from pathlib import Path

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import (
    LLMAssistant,
    get_config,
    LLMClient,
    EmbeddingClient,
    QdrantService,
    RAGService
)


async def example_parse_intent():
    """示例：解析用户设计意图"""
    print("=" * 60)
    print("示例 1: 解析设计意图")
    print("=" * 60)
    
    assistant = LLMAssistant()
    
    user_input = "我想设计一个工作在 1550nm 的光栅耦合器，效率最好能到 70% 以上"
    
    print(f"\n用户输入: {user_input}")
    print("\n正在解析...")
    
    intent = await assistant.parse_intent(user_input)
    
    print(f"\n解析结果:")
    print(f"  器件类型: {intent.device_type}")
    print(f"  目标规格: {intent.target_specs}")
    print(f"  置信度: {intent.confidence}")
    
    if intent.clarification_needed:
        print(f"  需要澄清: {intent.clarification_needed}")
    
    await assistant.close()
    return intent


async def example_generate_workflow(intent):
    """示例：生成工作流配置"""
    print("\n" + "=" * 60)
    print("示例 2: 生成工作流配置")
    print("=" * 60)
    
    assistant = LLMAssistant()
    
    print("\n正在生成工作流配置...")
    
    workflow = await assistant.generate_workflow(intent)
    
    print(f"\n推荐配置:")
    print(f"  管道名称: {workflow.pipeline_name}")
    print(f"  设计挑战: {workflow.challenge}")
    print(f"  模型配置: {workflow.model_config}")
    print(f"  选择理由: {workflow.rationale}")
    
    if workflow.alternatives:
        print(f"  备选方案: {workflow.alternatives}")
    
    await assistant.close()
    return workflow


async def example_explain_results():
    """示例：解释设计结果"""
    print("\n" + "=" * 60)
    print("示例 3: 解释设计结果")
    print("=" * 60)
    
    assistant = LLMAssistant()
    
    design_result = {
        "efficiency": 0.723,
        "bandwidth_nm": 95,
        "insertion_loss_db": 1.4
    }
    
    simulation_result = {
        "wavelength_nm": 1550,
        "peak_efficiency": 0.735,
        "1dB_bandwidth_nm": 85
    }
    
    print(f"\n设计结果: {design_result}")
    print(f"仿真结果: {simulation_result}")
    print("\n正在分析...")
    
    report = await assistant.explain_results(design_result, simulation_result)
    
    print(f"\n分析报告:")
    print(f"  摘要: {report.summary}")
    if report.suggestions:
        print(f"  建议: {report.suggestions}")
    
    await assistant.close()


async def example_chat():
    """示例：对话交互"""
    print("\n" + "=" * 60)
    print("示例 4: 对话交互")
    print("=" * 60)
    
    assistant = LLMAssistant()
    
    # 第一轮对话
    response1 = await assistant.chat("什么是光栅耦合器？")
    print(f"\n用户: 什么是光栅耦合器？")
    print(f"助手: {response1[:200]}...")
    
    # 第二轮对话（带上下文）
    history = [{"role": "user", "content": "什么是光栅耦合器？"}, {"role": "assistant", "content": response1}]
    response2 = await assistant.chat("它的典型效率是多少？", history=history)
    print(f"\n用户: 它的典型效率是多少？")
    print(f"助手: {response2[:200]}...")
    
    await assistant.close()


async def example_rag_indexing():
    """示例：索引知识库"""
    print("\n" + "=" * 60)
    print("示例 5: RAG 知识库索引")
    print("=" * 60)
    
    config = get_config()
    rag = RAGService(config)
    
    # 初始化向量数据库
    print("\n初始化 Qdrant...")
    success = rag.initialize(recreate=False)
    print(f"初始化{'成功' if success else '失败'}")
    
    # 索引知识库目录
    knowledge_dir = config.knowledge_dir
    if knowledge_dir.exists():
        print(f"\n索引知识库目录: {knowledge_dir}")
        count = await rag.index_directory(knowledge_dir, category="photonics")
        print(f"成功索引 {count} 个文件")
    else:
        print(f"知识库目录不存在: {knowledge_dir}")
    
    # 测试检索
    print("\n测试检索...")
    context = await rag.retrieve("光栅耦合器的效率", top_k=3)
    print(f"检索到 {len(context.documents)} 个相关文档")
    if context.documents:
        print(f"最相关文档 (分数: {context.documents[0].score:.2f}):")
        print(f"  {context.documents[0].content[:100]}...")


async def example_embedding():
    """示例：嵌入向量"""
    print("\n" + "=" * 60)
    print("示例 6: 文本嵌入")
    print("=" * 60)
    
    config = get_config()
    client = EmbeddingClient(config.embedding)
    
    texts = [
        "光栅耦合器是一种重要的光子学器件",
        "逆向设计使用神经网络自动优化设计参数"
    ]
    
    print(f"\n生成嵌入向量 (模型: {config.embedding.model})")
    
    async with client:
        results = await client.embed(texts)
        
        for i, result in enumerate(results):
            print(f"\n文本 {i+1}: {texts[i]}")
            print(f"  向量维度: {result.dimension}")
            print(f"  前5维: {result.embedding[:5]}")


async def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("HI-Photonics LLM 助手示例")
    print("=" * 60)
    
    # 检查配置
    config = get_config()
    print(f"\n配置状态:")
    print(f"  LLM 模型: {config.llm.model}")
    print(f"  LLM 已配置: {config.llm.is_configured}")
    print(f"  Embedding 模型: {config.embedding.model}")
    print(f"  Embedding 已配置: {config.embedding.is_configured}")
    print(f"  Qdrant 地址: {config.qdrant.url}")
    
    # 运行示例
    try:
        # 示例 1: 解析意图
        intent = await example_parse_intent()
        
        # 示例 2: 生成工作流
        workflow = await example_generate_workflow(intent)
        
        # 示例 3: 解释结果
        await example_explain_results()
        
        # 示例 4: 对话
        await example_chat()
        
        # 示例 5: RAG 索引（可选，需要 Qdrant 运行）
        # await example_rag_indexing()
        
        # 示例 6: 嵌入（可选，需要配置 Embedding API Key）
        # await example_embedding()
        
    except Exception as e:
        print(f"\n错误: {e}")
        print("请确保:")
        print("  1. 已在 .env 文件中配置 LLM_API_KEY")
        print("  2. 如需使用 Embedding，请配置 EMBEDDING_API_KEY")
        print("  3. 如需使用 RAG，请启动 Qdrant 服务")


if __name__ == "__main__":
    asyncio.run(main())
