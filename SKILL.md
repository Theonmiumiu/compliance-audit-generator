---
name: compliance-audit-generator
description: 自动基于多份合规文档生成检查工作底稿Excel文件，在用户提到"生成检查底稿"、"合规审查底稿"或类似任务时调用
---
# 技能概述
该技能用于自动化地接收同一合规领域的多份合规文件（外规+内规），提取条款、合并相似内容，生成符合客户模板格式的检查工作底稿Excel文件。

# 技能
请按照以下步骤完成任务：

## 1. 环境检查
检查 `/scripts` 文件夹中的 `.venv` 虚拟环境是否存在或可用。若不可用，请基于 `pyproject.toml` 和 `uv.lock` 创建虚拟环境。

## 2. 数据库初始化
运行 `/scripts/tools.py` 中的 `backfill_database_embeddings` 函数，确保数据库中的向量数据完整。

## 3. 生成底稿
运行 `/scripts/tools.py` 中的 `generate_audit_draft` 函数：
- 参数：`file_paths` - 合规文件路径列表（支持多个文件）
- 参数：`template_path` - 可选，自定义模板路径
- 返回：生成的Excel文件路径

## 4. 发送结果
将生成的Excel文件发送给用户。

# 函数解释

## backfill_database_embeddings(schema_name, table_name)
数据库向量补全函数，检查并补齐PostgreSQL RAG数据库中缺失的embedding。

## generate_audit_draft(file_paths, template_path=None)
主函数，生成检查工作底稿Excel文件。
- **输入**：多份合规文件路径列表
- **处理**：
  1. 解析所有文件，提取条款
  2. 使用语义相似度合并相似条款
  3. 为每条检查内容生成检查依据、检查方式、对应材料
  4. 输出Excel文件
- **输出**：Excel文件路径

# 输出格式
Excel文件包含以下列：
| 序号 | 检查内容 | 检查依据 | 检查方式 | 对应材料 | 检查发现问题 | 备注 |

- **检查内容**：条款的改写/摘要（非原文）
- **检查依据**：文件名 + 条款号（相似条款列出多个来源）
- **检查方式**：具体检查操作步骤
- **对应材料**：从5大类材料中选择

# 注意事项
1. 支持 .pdf、.docx、.doc 格式的文件
2. 条款合并基于语义相似度，由LLM判断
3. 对应材料从预定义的5大类材料中选择
4. 严禁读取 .env 文件中的敏感信息