package cn.junlaile.bert.chunker.pojo;

/**
 * 特殊分词
 * @param content 特殊token的字符串表示
 * @param lstrip 是否在token左侧去除空格 false 表示不处理
 * @param rstrip 是否在token右侧去除空格 false 表示不处理
 * @param normalized 是否对token进行规范化 false 表示不处理
 * @param singleWord 是否要求token是一个单词 false 表示不限制
 */
public record SpecialToken(String content, boolean lstrip, boolean rstrip, boolean normalized, boolean singleWord){
}
