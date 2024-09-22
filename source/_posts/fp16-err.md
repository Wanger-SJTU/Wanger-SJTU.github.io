---
title: fp16 的累加误差有多大
tags:
  - DL
  - 数值精度
category:
  - 技术
date: 2024-09-22 14:41:48
---
最近在项目中需要实现fp16的数据类型做FFN的计算，算子实现的同学反馈误差与x86上得到的golden数据有比较大误差。开始以为是x86侧做数值模拟仿真的问题。后面也实现了对比了一下，发现误差累计确实挺大。

## 实测结果对比
```c++
int main()
{
    // Seed with a real random value, if available
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 0.01);
    
    float16_t lhs[4096] = {0};
    float16_t rhs[4096] = {0};
    for (int i = 0; i < 4096; i++) {
        lhs[i] =  dist(gen);
        rhs[i] =  dist(gen);
    }
    float16_t res_fp16 = 0;
    float res_fp32 = 0;

    for (int i = 0; i < 4096; i++) {
        res_fp16 += lhs[i] * rhs[i];
        res_fp32 += lhs[i] * rhs[i];
    }
    std::cout << "fp16 " << res_fp16 << std::endl;
    std::cout << "fp32 " << res_fp32 << std::endl;
    wirte2file("/data/local/tmp/lhs", reinterpret_cast<char*>(lhs), 8192);
    wirte2file("/data/local/tmp/rhs", reinterpret_cast<char*>(rhs), 8192);
}
```
结果输出：
```
fp16 0.0942383
fp32 0.103176
```
相对误差到8.1%了。难怪反馈有问题。

| dim  | 绝对误差         |
| ---- | ------------ |
| 100  | 1.63913e-07  |
| 1000 | -0.00033829  |
| 2000 | -0.000909835 |
| 4000 | -0.00924221  |
## golden 数据误差从何而来
实际生成golden数据的时候，也考虑了数值类型差异的影响，那为什么还存在误差呢？
    
> 对比了一下dot的视线与直接累加结果

```python
import numpy as np
import torch

lhs = np.fromfile("lhs",dtype=np.float16)
rhs = np.fromfile("rhs",dtype=np.float16)

lhs = torch.from_numpy(lhs)
rhs = torch.from_numpy(rhs)

res = torch.Tensor([1]).half()
res[0] = 0
for i in range(4096):
    res += lhs[i:i+1] * rhs[i:i+1]

print(res)
print(torch.dot(lhs, rhs))
```

```
tensor([0.0942], dtype=torch.float16)
tensor(0.1041, dtype=torch.float16)
```
结果对得上了。torch 的 dot实现的时候很可能用了更高数值类型做累加。