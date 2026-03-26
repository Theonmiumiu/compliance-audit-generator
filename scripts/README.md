# 合规检查工作底稿自动生成工具

## 概述

本工具用于自动化地接收同一合规领域的多份合规文件（外规+内规），提取条款、合并相似内容，生成符合客户模板格式的检查工作底稿Excel文件。

## 功能特点

- 支持多份文件输入（PDF、DOCX、DOC）
- 基于语义相似度合并相似条款
- 自动生成检查内容、检查方式、对应材料
- 输出标准格式的Excel检查工作底稿

## 使用方法

```python
from tools import generate_audit_draft

# 输入多份文件
files = [
    '文件1_外规.pdf',
    '文件2_内规.docx'
]

# 生成底稿
result = generate_audit_draft(files)
print(f"生成完成: {result}")
```