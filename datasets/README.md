# datasets/

放置真实数据集 CSV（如 5G/LTE 移动通信流量、NSL-KDD、CICIDS 风格），并在
`configs/meta_train.yaml` 的 `data.csv_path` 指向它。CSV 需包含一个标签列
（默认列名 `label`），其余为特征列（数值列自动标准化、字符串/低基数列自动 One-Hot）。

若该路径文件不存在且 `data.synthetic_if_missing: true`，系统会自动生成一个带类间
可分性与时序结构的合成移动通信流量数据集 `mobile_traffic.csv` 以跑通完整流程。
也可手动生成：

```bash
python scripts/generate_synthetic_data.py --n-per-class 1000
```
