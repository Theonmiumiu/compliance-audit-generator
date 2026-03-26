# 条款相似度合并功能设计方案

**日期**: 2026-03-19
**功能**: 多文件条款相似度检测与合并

---

## 1. 需求背景

项目入口接收多份合规文件，需要将条款切分后比较是否有极其相似的条款（来自不同文件）。若相似度大于阈值（0.85），则在最终 Excel 中合并为一条记录，共享检查内容和检查方式，同时在检查依据列列出所有来源。

---

## 2. 整体流程

```
[多份文件]
    ↓
[条款切分] (已有逻辑 chunk_regulation_text)
    ↓
[Embedding 粗筛] (top_K=6, similarity≥0.75)
    ↓
[Rerank 精确比较] (threshold=0.85)
    ↓
[分组合并] (只合并两两相似，不做链式)
    ↓
[后续 LLM 生成检查内容] (已有逻辑)
    ↓
[Excel 输出] (检查依据列列出所有来源)
```

---

## 3. 核心逻辑

### 3.1 分组合并算法 (`_merge_similar_articles`)

输入：所有条款列表 `articles`

输出：合并后的条款列表，每个元素包含 `content` 和 `sources`

**算法步骤**：

1. 对所有条款生成 Embedding 向量
2. 对每条条款 i：
   - 使用向量相似度从其他条款中粗筛 top_K=6 条（相似度 ≥ 0.75）
   - 将条款 i 与粗筛结果一起调用 Rerank API
   - Rerank 返回每条候选的精确相似度分数
   - 分数 > 0.85 的条款归入当前组合
   - 记录所有来源的元数据
3. 重复步骤 2 直到处理完所有条款

**不链式合并原则**：
- A≈B 且 B≈C 但 A≠C → A、B、C 不自动合并为一组
- B 可能同时出现在 A 的组合和 C 的组合中

### 3.2 元数据传递

合并组携带 `sources: [metadata_A, metadata_B, ...]`
- `metadata` 包含 `doc_name`, `article_num`, `chapter` 等

### 3.3 检查依据格式化

已有函数 `_format_sources()` 负责将 `sources` 渲染为检查依据列：
```
《证券期货经营机构廉洁从业规定》第六条；《期货经营机构廉洁从业实施细则》第五条
```

---

## 4. 性能优化策略

### 4.1 Embedding 粗筛

- 使用 numpy 向量矩阵计算，一次性计算所有条款对的余弦相似度
- 对每条条款取 top_K=6 最相似的候选进入 Rerank
- 过滤掉相似度 < 0.75 的候选

### 4.2 Rerank 调用优化

- 每条条款最多只做一次 batch rerank 调用
- 总调用次数 = 条款数，而不是 n(n-1)/2
- API 支持 top_n 参数设大值，一次返回所有候选分数

---

## 5. 模块改动

### 5.1 `tools.py` 改动

**新增函数**：
- `_compute_embeddings()`: 批量生成条款 Embedding
- `_embedding_similarity()`: 向量相似度计算
- `_batch_rerank()`: 批量 Rerank 调用

**修改函数**：
- `_merge_similar_articles()`: 重写为两阶段相似度合并逻辑

### 5.2 `src/utils/models.py` 改动

**新增方法**：
- `ModelFactory.get_embedding_model()`: 获取 Embedding 模型（用于粗筛）

### 5.3 `src/utils/rag.py` 改动

**复用已有**：
- `RAGModelServer.get_embeddings()`: 批量向量化（已有）
- `RAGModelServer.rerank_documents()`: Rerank 精排（已有，需扩展支持返回分数）

---

## 6. 数据结构

### 6.1 条款数据结构

```python
# 输入
article = {
    "content": "条款原文",
    "metadata": {
        "doc_name": "文件名",
        "article_num": "第X条",
        "chapter": "章节名"
    }
}

# 合并后输出
merged_article = {
    "content": "条款原文（合并后）",
    "sources": [
        {"doc_name": "文件A", "article_num": "第六条"},
        {"doc_name": "文件B", "article_num": "第五条"}
    ]
}
```

### 6.2 Rerank 返回扩展

扩展 `RAGModelServer.rerank_documents()` 支持返回分数：
```python
def rerank_documents_with_scores(self, query: str, documents: List[str], top_n: int) -> List[Tuple[str, float]]:
    """返回重排序后的文档及对应分数"""
    ...
```

---

## 7. 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `EMBEDDING_TOP_K` | 6 | Embedding 粗筛候选数 |
| `EMBEDDING_THRESHOLD` | 0.75 | Embedding 粗筛相似度阈值 |
| `RERANK_THRESHOLD` | 0.85 | Rerank 合并阈值 |

---

## 8. 边界处理

1. **单文件输入**: 跳过合并步骤
2. **无相似条款**: 每条条款独立，不合并
3. **候选不足 6 条**: 全部送入 Rerank
4. **Rerank API 失败**: 降级为使用 Embedding 相似度结果

---

## 9. 测试计划

1. 单元测试：`_embedding_similarity()` 向量计算正确性
2. 集成测试：两文件输入，验证相似条款被正确合并
3. 性能测试：100 条条款的合并耗时 < 5秒（不含 LLM 调用）
