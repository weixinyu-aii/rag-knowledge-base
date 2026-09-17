# 检索评测集

questions.jsonl 每行包含：

- id：问题编号
- question：测试问题
- relevant_sources：相关文档名
- relevant_pages：可选，限定相关页码
- relevant_terms：可选，要求文本块同时包含指定词项，用于块级相关性判断
- expected_keywords：可选，用于答案关键词覆盖率评测

运行检索评测：

~~~bash
python -m rag_knowledge_base evaluate --dataset examples/evaluation/questions.jsonl --k 3
~~~

运行端到端答案评测：

~~~bash
python -m rag_knowledge_base evaluate --dataset examples/evaluation/questions.jsonl --k 3 --with-answers
~~~
