# BERT Chunker for Chinese (Java Implementation)

这是 [tim1900/bert-chunker-Chinese-2](https://hf-mirror.com/tim1900/bert-chunker-Chinese-2) 的Java实现版本，提供了基于BERT模型的中文文本智能分块功能。本项目利用ONNX Runtime进行模型推理，实现了高效的中文文本分块处理。

## 功能介绍

- 智能文本分块：根据语义边界自动将长文本分割成合适的块
- 可调节的分块粒度：通过概率阈值参数控制分块的细粒度
- 适用于中文文本：针对中文语言特点进行优化
- 基于ONNX：使用ONNX Runtime高效执行BERT模型推理
- Java原生实现：便于集成到Java项目中

## 环境要求

- Java 21
- Maven 3.6+

## 依赖库

```xml
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime</artifactId>
    <version>1.21.0</version>
</dependency>
<dependency>
    <groupId>com.alibaba.fastjson2</groupId>
    <artifactId>fastjson2</artifactId>
    <version>2.0.45</version>
</dependency>
<dependency>
    <groupId>org.slf4j</groupId>
    <artifactId>slf4j-api</artifactId>
    <version>2.0.9</version>
</dependency>
```

## 快速开始

```java
import cn.junlaile.bert.chunker.token.BertChunkerInference;
import java.util.List;

public class Example {
    public static void main(String[] args) {
        // 加载词汇表和特殊标记
        BertChunkerInference.loadVocab();
        BertChunkerInference.loadSpecialTokens();
        
        // 待分块的长文本
        String text = """
                这是一段需要分块的长文本。文本可以包含多个段落和句子。
                智能分块功能会根据语义边界自动划分合适的文本块。
                这对于处理长文本时非常有用，可以使模型更好地理解上下文。
                """;
        
        // 设置概率阈值(0.0-1.0)，值越小分块越细
        float probThreshold = 0.5f;
        
        // 执行文本分块
        List<String> chunks = BertChunkerInference.chunkText(text, probThreshold);
        
        // 处理分块结果
        for (int i = 0; i < chunks.size(); i++) {
            System.out.println("===== 块 " + i + " =====");
            System.out.println(chunks.get(i));
        }
    }
}
```

## 核心API说明

### BertChunkerInference

这是实现文本分块功能的主类，提供以下主要方法：

```java
// 加载词汇表
public static void loadVocab()

// 加载特殊标记
public static void loadSpecialTokens()

// 将文本转换为模型输入（token_ids, attention_mask, token_type_ids）
public static Map<String, long[]> tokenize(String text)

// 文本分块
// @param text 需要分块的文本
// @param probThreshold 分块细分度，值越小产生的块越多(0.0-1.0)
// @return 分块后的文本列表
public static List<String> chunkText(String text, float probThreshold)
```

## 技术实现

1. **文本预处理**：将输入文本转换为BERT模型所需的token序列
2. **滑动窗口处理**：对于长文本，采用滑动窗口方式进行分块处理
3. **智能分块点判断**：基于BERT模型输出的概率值，确定适合的分块位置
4. **参数控制**：通过probThreshold参数控制分块的细粒度

## 资源文件

项目依赖以下资源文件（位于classpath的/onnx/目录下）：

- model.onnx：BERT分块模型
- vocab.txt：BERT词汇表
- special_tokens_map.json：特殊标记定义

## 与原始项目的区别

本项目是[tim1900/bert-chunker-Chinese-2](https://hf-mirror.com/tim1900/bert-chunker-Chinese-2)的Java实现版本，核心功能相同，但使用了Java生态系统的工具和库。采用ONNX Runtime直接进行模型推理，提供了与原Python版本相同的分块效果，但更易于集成到Java项目中。

## 许可证

与原项目保持一致的许可证。

## 致谢

- 感谢[tim1900](https://hf-mirror.com/tim1900)提供的原始BERT中文分块模型
- [Hugging Face](https://huggingface.co/)提供的模型托管服务