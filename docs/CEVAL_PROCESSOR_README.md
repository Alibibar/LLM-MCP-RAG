# C-Eval Dataset Processor

A robust Python script for processing C-Eval dataset parquet files and merging validation data into a single JSON file with comprehensive error handling.

## 背景 (Background)

C-Eval是一个全面的中文基础模型评估套件，包含了13,948个多项选择题，涵盖52个不同学科和四个难度级别。数据集按科目分类，每个科目包含dev、val和test三部分，以parquet格式存储。

C-Eval is a comprehensive Chinese benchmark for foundation models, containing 13,948 multiple-choice questions across 52 subjects and four difficulty levels. The dataset is organized by subject, with each subject containing dev, val, and test splits stored in parquet format.

## 主要特性 (Key Features)

- **可靠的Parquet读取**: 使用PyArrow直接读取parquet文件，避免pandas兼容性问题
- **智能错误处理**: 多种读取策略处理损坏或不兼容的parquet文件
- **进度跟踪**: 详细的进度显示和日志记录
- **数据完整性**: 为每条数据添加科目字段，保持数据来源可追溯
- **灵活输出**: 包含元数据的结构化JSON输出

- **Reliable Parquet Reading**: Uses PyArrow directly to avoid pandas compatibility issues
- **Smart Error Handling**: Multiple reading strategies for corrupted or incompatible parquet files
- **Progress Tracking**: Detailed progress display and logging
- **Data Integrity**: Adds subject field to each record for traceability
- **Flexible Output**: Structured JSON output with metadata

## 安装要求 (Requirements)

```bash
pip install pyarrow
```

## 使用方法 (Usage)

### 基本用法 (Basic Usage)

```bash
python src/ceval_processor.py --data_dir /path/to/ceval --output ceval_val_merged.json
```

### 完整参数 (Full Parameters)

```bash
python src/ceval_processor.py \
    --data_dir ./ceval_data \
    --output ./output/ceval_validation.json \
    --log_level INFO
```

### 参数说明 (Parameter Description)

- `--data_dir, -d`: C-Eval数据集目录路径（包含各科目子目录）
- `--output, -o`: 输出JSON文件路径
- `--log_level, -l`: 日志级别 (DEBUG, INFO, WARNING, ERROR)

## 数据集结构 (Dataset Structure)

期望的C-Eval数据集目录结构：

Expected C-Eval dataset directory structure:

```
ceval_data/
├── computer_science/
│   ├── dev.parquet
│   ├── val.parquet
│   └── test.parquet
├── mathematics/
│   ├── dev.parquet
│   ├── val.parquet
│   └── test.parquet
├── physics/
│   └── ...
└── ...
```

## 输出格式 (Output Format)

输出JSON文件包含以下结构：

The output JSON file contains the following structure:

```json
{
  "metadata": {
    "total_records": 1346,
    "processed_subjects": 52,
    "failed_subjects": 0,
    "subjects_list": ["computer_science", "mathematics", "physics", "..."],
    "failed_subjects_list": [],
    "dataset": "C-Eval",
    "split": "validation"
  },
  "data": [
    {
      "id": 1,
      "question": "问题内容",
      "A": "选项A",
      "B": "选项B", 
      "C": "选项C",
      "D": "选项D",
      "answer": "A",
      "subject": "computer_science"
    },
    ...
  ]
}
```

## 错误处理 (Error Handling)

脚本实现了多层错误处理机制：

The script implements multi-level error handling:

### Parquet读取错误 (Parquet Reading Errors)

当遇到 `OSError: Repetition level histogram size mismatch` 或其他parquet读取错误时：

When encountering `OSError: Repetition level histogram size mismatch` or other parquet reading errors:

1. **主要策略**: 使用PyArrow的标准读取方法
2. **备用策略**: 使用PyArrow的legacy模式读取
3. **失败处理**: 记录错误并继续处理其他文件

1. **Primary Strategy**: Use PyArrow's standard reading method
2. **Fallback Strategy**: Use PyArrow's legacy mode reading
3. **Failure Handling**: Log errors and continue processing other files

### 文件查找 (File Discovery)

- 智能查找validation文件（支持多种命名模式）
- 自动跳过缺失的科目目录
- 详细的错误报告

- Smart validation file discovery (supports multiple naming patterns)
- Automatic skipping of missing subject directories  
- Detailed error reporting

## 日志记录 (Logging)

脚本会生成两种日志输出：

The script generates two types of log output:

1. **控制台输出**: 实时进度和重要信息
2. **日志文件**: 详细的处理日志保存到 `ceval_processor.log`

1. **Console Output**: Real-time progress and important information
2. **Log File**: Detailed processing logs saved to `ceval_processor.log`

### 日志级别 (Log Levels)

- `DEBUG`: 详细的调试信息，包括每个文件的读取状态
- `INFO`: 处理进度和重要信息（推荐）
- `WARNING`: 警告和非致命错误
- `ERROR`: 严重错误信息

- `DEBUG`: Detailed debugging information including file reading status
- `INFO`: Processing progress and important information (recommended)
- `WARNING`: Warnings and non-fatal errors
- `ERROR`: Critical error information

## 性能优化 (Performance Optimization)

- **内存效率**: 流式处理大文件，避免内存溢出
- **并发安全**: 单线程处理确保数据一致性
- **错误恢复**: 单个文件失败不影响整体处理

- **Memory Efficient**: Stream processing of large files to avoid memory overflow
- **Concurrency Safe**: Single-threaded processing ensures data consistency
- **Error Recovery**: Single file failures don't affect overall processing

## 故障排除 (Troubleshooting)

### 常见问题 (Common Issues)

#### 1. PyArrow未安装 (PyArrow Not Installed)
```bash
Error: PyArrow is required. Install it with: pip install pyarrow
```
**解决方案**: 运行 `pip install pyarrow`

#### 2. 数据目录不存在 (Data Directory Not Found)
```bash
FileNotFoundError: Data directory not found: /path/to/data
```
**解决方案**: 检查数据目录路径是否正确

#### 3. 没有找到科目目录 (No Subject Directories Found)
```bash
ValueError: No subject directories found in the dataset
```
**解决方案**: 确保数据目录包含带有parquet文件的子目录

#### 4. Parquet读取失败 (Parquet Reading Failed)
```bash
PyArrow failed for file.parquet: OSError: Repetition level histogram size mismatch
```
**解决方案**: 脚本会自动尝试备用读取方法，但如果文件完全损坏，会被跳过并记录

### 调试技巧 (Debugging Tips)

1. **使用DEBUG日志级别**:
   ```bash
   python src/ceval_processor.py --data_dir ./data --output result.json --log_level DEBUG
   ```

2. **检查日志文件**:
   ```bash
   tail -f ceval_processor.log
   ```

3. **验证输出**:
   ```bash
   python -c "import json; data = json.load(open('result.json')); print(f'Records: {data[\"metadata\"][\"total_records\"]}')"
   ```

## 扩展功能 (Extension Features)

### 自定义字段添加 (Custom Field Addition)

可以修改 `process_subject` 方法来添加自定义字段：

You can modify the `process_subject` method to add custom fields:

```python
# Add custom fields to each record
for record in records:
    record['subject'] = subject_name
    record['source'] = 'ceval'
    record['processed_at'] = datetime.now().isoformat()
```

### 并行处理 (Parallel Processing)

对于大型数据集，可以考虑添加多进程支持：

For large datasets, consider adding multiprocessing support:

```python
from multiprocessing import Pool

def process_subjects_parallel(self, subjects):
    with Pool() as pool:
        results = pool.map(self.process_subject, subjects)
    return [r for r in results if r is not None]
```

## 许可证 (License)

本脚本遵循项目的许可证条款。

This script follows the project's license terms.

## 贡献 (Contributing)

欢迎提交问题报告和改进建议。

Issue reports and improvement suggestions are welcome.