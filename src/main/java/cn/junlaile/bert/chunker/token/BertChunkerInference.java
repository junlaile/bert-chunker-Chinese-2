package cn.junlaile.bert.chunker.token;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import cn.junlaile.bert.chunker.pojo.SpecialToken;
import com.alibaba.fastjson2.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * bert分块实现
 */
public class BertChunkerInference {

    private static final Logger logger = LoggerFactory.getLogger(BertChunkerInference.class);

    private static final Map<String, Integer> VOCAB_MAP = new HashMap<>();

    private static final Map<String, SpecialToken> SPECIAL_TOKENS_MAP = new HashMap<>();

    // 在BertChunkerInference类中添加一个静态Map用于存储当前处理文本的token到字符位置的映射
    private static final ThreadLocal<Map<Integer, Integer>> TOKEN_TO_CHAR_MAP =
            ThreadLocal.withInitial(HashMap::new);

    private static final String VOCAB_PATH = "/onnx/vocab.txt";
    private static final String SPECIAL_TOKENS_PATH = "/onnx/special_tokens_map.json";
    public static final String MODEL_PATH = "/onnx/model.onnx";

    public static final int MAX_LENGTH = 512;

    /**
     * 加载VOCAB词汇表
     */
    public static void loadVocab() {
        try (InputStream inputStream = BertChunkerInference.class.getResourceAsStream(VOCAB_PATH)) {
            if (Objects.isNull(inputStream)){
                throw new IOException("未读取到"+VOCAB_PATH);
            }
            try (BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8))){
                String line;
                int id = 0;
                while ((line = bufferedReader.readLine()) != null) {
                    VOCAB_MAP.put(line.trim(), id++);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException("无法加载词汇表: " + e.getMessage(), e);
        }
        logger.info("Vocab loaded with {} tokens.", VOCAB_MAP.size());
    }

    /**
     * 加载 special_tokens_map.json
     */
    public static void loadSpecialTokens() {
        String jsonString;
        try (InputStream inputStream = BertChunkerInference.class.getResourceAsStream(SPECIAL_TOKENS_PATH)) {
            if (inputStream == null) {
                throw new IOException("未找到特殊Token资源: " + SPECIAL_TOKENS_PATH);
            }
            jsonString = new String(inputStream.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new RuntimeException("无法加载特殊Token: " + e.getMessage(), e);
        }
        JSONObject jsonObject = JSONObject.parseObject(jsonString);
        for (Map.Entry<String, Object> entry : jsonObject.entrySet()) {
            String key = entry.getKey();
            SpecialToken specialToken = jsonObject.getJSONObject(key).toJavaObject(SpecialToken.class);
            SPECIAL_TOKENS_MAP.put(key, specialToken);
        }
        // 验证必要 token 存在
        if (!SPECIAL_TOKENS_MAP.containsKey("cls_token") ||
                !SPECIAL_TOKENS_MAP.containsKey("mask_token") ||
                !SPECIAL_TOKENS_MAP.containsKey("pad_token") ||
                !SPECIAL_TOKENS_MAP.containsKey("sep_token")) {
            throw new RuntimeException("special_tokens_map.json is missing required tokens!");
        }
        for (SpecialToken specialToken : SPECIAL_TOKENS_MAP.values()) {
            if (!VOCAB_MAP.containsKey(specialToken.content())) {
                throw new RuntimeException("Special token " + specialToken.content() + " not found in vocab.txt");
            }
        }
        logger.info("Special tokens loaded:{}", SPECIAL_TOKENS_MAP.keySet());
    }

    /**
     * 将文本转换为 token_ids, attention_mask 和 token_type_ids
     */
    public static Map<String, long[]> tokenize(String text) {
        Map<Integer, Integer> tokenToCharMap = TOKEN_TO_CHAR_MAP.get();
        // 清除之前的映射
        tokenToCharMap.clear();

        List<String> tokenList = new ArrayList<>();
        SpecialToken clsToken = SPECIAL_TOKENS_MAP.get("cls_token");
        SpecialToken sepToken = SPECIAL_TOKENS_MAP.get("sep_token");
        SpecialToken unkToken = SPECIAL_TOKENS_MAP.get("unk_token");
        //添加起始标记
        tokenList.add(clsToken.content());
        // CLS标记不对应原文中的位置
        tokenToCharMap.put(0, -1);

        // 处理文本中的每个字符
        int charPos = 0;
        for (char c : text.toCharArray()) {
            // 应用 lstrip rstrip normalized 默认都为false
            tokenList.add(String.valueOf(c));
            // 存储token索引到字符位置的映射
            tokenToCharMap.put(tokenList.size() - 1, charPos);
            charPos++;
        }
        //添加结束标记
        tokenList.add(sepToken.content());
        // SEP标记不对应原文中的位置
        tokenToCharMap.put(tokenList.size() - 1, -1);

        int effectiveLength = Math.min(tokenList.size(), MAX_LENGTH);
        long[] inputIds = new long[MAX_LENGTH];
        long[] attentionMask = new long[MAX_LENGTH];
        long[] tokenTypeIds = new long[MAX_LENGTH];

        int i = 0;
        int unkId = VOCAB_MAP.get(unkToken.content());
        int padId = VOCAB_MAP.get(SPECIAL_TOKENS_MAP.get("pad_token").content());

        for (; i < effectiveLength; i++) {
            String token = tokenList.get(i);
            inputIds[i] = VOCAB_MAP.getOrDefault(token, unkId);
            attentionMask[i] = 1;
            tokenTypeIds[i] = 0;
        }

        for (; i < MAX_LENGTH; i++) {
            inputIds[i] = padId;
            attentionMask[i] = 0;
            tokenTypeIds[i] = 0;
        }
        HashMap<String, long[]> inputs = new HashMap<>();
        inputs.put("input_ids", inputIds);
        inputs.put("attention_mask", attentionMask);
        inputs.put("token_type_ids", tokenTypeIds);
//        inputs.put("input.3",tokenTypeIds);

        return inputs;
    }

    /**
     * 文本分块
     *
     * @param text          文本
     * @param probThreshold 分块细分度，值越小产生的块越多(0.0-1.0)
     * @return 分块后的语句
     */
    public static List<String> chunkText(String text, float probThreshold) {
        List<Integer> splitPositions = new ArrayList<>();

        // 计算logits阈值 (对应Python中的math.log(1/prob_threshold - 1))
        float logitsThreshold = (float) Math.log(1 / probThreshold - 1);

        // 估计token数量
        int totalLength = text.length();

        logger.info("处理约 {} 个字符...", totalLength);

        // 初始化滑动窗口
        int windowsStart = 0;

        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        try (InputStream modelStream = BertChunkerInference.class.getResourceAsStream(BertChunkerInference.MODEL_PATH)) {
            if (modelStream == null) {
                throw new IOException("未找到模型资源: " + BertChunkerInference.MODEL_PATH);
            }
            // 加载模型字节数组
            byte[] modelBytes = modelStream.readAllBytes();
            try (OrtEnvironment env = OrtEnvironment.getEnvironment();
                 OrtSession session = env.createSession(modelBytes, options)) {

                while (windowsStart < totalLength) {
                    // 计算当前窗口的结束位置，确保不超出文本长度
                    int windowEnd = Math.min(totalLength, windowsStart + MAX_LENGTH);
                    // 获取当前窗口的文本
                    String windowText = text.substring(windowsStart, windowEnd);
                    // 获取当前窗口的文本
                    Map<String, long[]> inputs = tokenize(windowText);

                    // 执行推理
                    HashMap<String, OnnxTensor> inputMap = new HashMap<>();
                    for (Map.Entry<String, long[]> entry : inputs.entrySet()) {
                        String name = entry.getKey();
                        long[] data = entry.getValue();
                        OnnxTensor tensor = OnnxTensor.createTensor(env, LongBuffer.wrap(data),
                                new long[]{1, MAX_LENGTH});
                        inputMap.put(name, tensor);
                    }

                    try (OrtSession.Result result = session.run(inputMap)) {
                        float[][][] logits = (float[][][]) result.get(0).getValue();

                        List<Integer> windowSplitPositions = new ArrayList<>();
                        // 查找当前窗口的分块点
                        for (int i = 1; i < logits[0].length - 1; i++) {
                            if (logits[0][i][1] > (logits[0][i][0] - logitsThreshold)) {
                                int charPos = tokenToCharPosition(i);
                                if (charPos > 0) {
                                    windowSplitPositions.add(charPos + windowsStart);
                                }
                            }
                        }

                        // 处理窗口滑动逻辑
                        if (!windowSplitPositions.isEmpty()) {
                            splitPositions.addAll(windowSplitPositions);
                            // 以最后一个分块点为起点滑动窗口
                            int lastCharPos = windowSplitPositions.getLast() - windowsStart;
                            windowsStart += lastCharPos;
                        } else {
                            // 如果没有找到分块点，移动一个较小的步长，确保不会超出文本长度
                            int step = Math.min(MAX_LENGTH / 2, totalLength - windowsStart);
                            // 确保至少前进1个字符，避免死循环 Math.max(step, 1)
                            windowsStart += step;
                        }
                    }
                }
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // 根据分块点分割文本
        List<String> chunks = new ArrayList<>();
        int startPos = 0;
        for (int pos : splitPositions) {
            chunks.add(text.substring(startPos, pos));
            startPos = pos;
        }
        // 添加最后一个块
        if (startPos < text.length()) {
            chunks.add(text.substring(startPos));
        }

        return chunks;
    }

    // 使用映射关系的tokenToCharPosition方法
    public static int tokenToCharPosition(int tokenIndex) {
        Map<Integer, Integer> tokenToCharMap = TOKEN_TO_CHAR_MAP.get();
        return tokenToCharMap.getOrDefault(tokenIndex, -1);
    }


}
