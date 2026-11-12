# 积水观测的显式持续时间状态推断

这是独立的防汛演练资料Python项目。`data/`保存虚构人数、观测、时序或方案，`response_data.io`提供严格数值读取与来源字节SHA-256，`response_data.domain`给出已有原始摘要，`response_data.cli`输出JSON。字段、量纲与分析口径见[资料约定](docs/data-contract.md)。这些数据用于离线复盘，不连接真实人员、传感器或指挥系统。

Python 3.11及以上即可直接使用，无第三方Python依赖。在项目根目录执行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q response_data tests
python -m response_data.cli inspect data/normal.json
python -m response_data.cli infer data/normal.json
python -m response_data.cli score data/normal.json '[{"state":"dry","length":2},{"state":"rising","length":2},{"state":"holding","length":2},{"state":"falling","length":1}]'
```

`inspect`显示原始资料概况，不覆盖源文件。`infer`按冻结约定做显式持续时间前向／后向推断：资料似然、逐时隙状态后验、边界转移后验、最优分段路径与全枚举核对，方法、输出字段与长缺测段／尾窗截断／全量程上限／不可能持续长度四类边界情形见[推断说明](docs/inference.md)。`score`从逐段转移、持续概率和逐观测项复算一条给定路径的得分。`invalid-`前缀资料为不合法输入，其余变体使用相同字段保留不同业务边界。数值null须按字段语义区别于0，不在读取时自动填补。

Dockerfile固定基础镜像，可用`docker build -t response-data .`及`docker run --rm response-data`读取资料；`docker run --rm response-data python -m unittest discover -s tests -v`执行已有检查。运行所需资料全部在本地，无需登录或实时来源。
