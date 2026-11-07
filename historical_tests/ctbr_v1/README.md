# CTBR v1 历史测试

这些测试对应 main 的基准提交 6e059b1af2aa057e486dfb109e4d9c218c1def6f。

该实现采用 7 个成本输出、15 次物理转移、IPC 前端、参考调节器与制动恢复。
新主线采用 28N 成本、原生 PathCover 和独立 CTBR 观测，因此这些接口断言已失效。
文件保留原样，复现时应检出上述提交。当前默认 pytest 只收集 tests/。

当前测试位于 tests/test_acmpc_node_costs.py。原作者前向 fixture、NumPy
动力学对照、雅可比和 Hessian 测试继续位于 tests/test_turbompc_ctbr.py，
其测试主体未改动。可选 KKT 精修器的独立数学测试仍保留在 tests/。
