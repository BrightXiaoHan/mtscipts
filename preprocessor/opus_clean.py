import typer
import yaml
import os
import tempfile
import shutil

from opusfilter.opusfilter import OpusFilter
from opusfilter.util import yaml


RULES_CONFIG = """
common:
  output_directory: {folder}
  chunksize: 1000000
steps:

  # Step 1
  - type: remove_duplicates
    parameters:
      inputs: [{src_lang}, {trg_lang}]
      outputs: [{src_lang}.dedup, {tgt_lang}.dedup]

  # Step 2
  - type: filter
    parameters:
      inputs: [{src_lang}.dedup, {tgt_lang}.dedup]
      outputs: [{src_lang}.rules, {tgt_lang}.rules]
      filters:
        - LengthFilter:
            unit: [{src_unit}, {tgt_unit}]
            min_length: 1
            max_length: 150

        - HtmlTagFilter: {}

        - CharacterScoreFilter:
            scripts: [{src_char}, {tgt_char}]
            thresholds: [0.5, 0.8]
 
  # Step 3
  - type: train_alignment
    parameters:
      src_data: {src_lang}.rules
      tgt_data: {tgt_lang}.rules
      parameters:
        model: 3
        src_tokenizer: [{src_tokenizer}, {src_lang}]
        tgt_tokenizer: [{tgt_tokenizer}, {tgt_lang}]
      scores: align_score.jsonl
      output: align.priors
"""

ALINGMENT_CONFIG = """
common:
  output_directory: {folder}
  chunksize: 1000000
steps:
  - type: score
    parameters:
      inputs: [{src_lang}.rules, {tgt_lang}.rules]
      output: align_score.jsonl
      filters: &scorefilt
        - WordAlignFilter:
            src_threshold: 0 
            tgt_threshold: 0
            model: 3
            priors: align.priors
            src_tokenizer: [{src_tokenizer}, {src_lang}]
            tgt_tokenizer: [{tgt_tokenizer}, {tgt_lang}]

  - type: sort
    parameters:
      inputs:
        - {src_lang}.rules
        - {tgt_lang}.rules
      outputs:
        - {src_lang}.align
        - {tgt_lang}.align
      values: align_score.jsonl
      key:
        - WordAlignFilter.0
        - WordAlignFilter.1
      combine_operator: add

  - type: head
    parameters:
      inputs: [zh.align, en.align]
      outputs: [zh.final, en.final]
      n: {align_filter_remain}
"""

UNIT_MAPPING = {
    "zh": "char",
    "ja": "char",
    "en": "word",
    "fr": "word",
    "de": "word",
    "es": "word",
    "ru": "word",
    "ar": "word",
    "pt": "word",
}

CHARACTER_MAPPING = {
    "zh": "Han",
    "ja": "Hira",
    "en": "Latin",
    "fr": "Latin",
    "de": "Latin",
    "es": "Latin",
    "ru": "Cyrl",
    "ar": "Arab",
    "pt": "Latin",
}

TOKENIZER_MAPPING = {
    "zh": "jieba",
    "ja": "mecab",
    "en": "moses",
    "fr": "moses",
    "de": "moses",
    "es": "moses",
    "ru": "moses",
    "ar": "moses",
    "pt": "moses",
}


def opus_clean(
    folder: str = typer.Argument(..., help="存放语料的文件夹"),
    src_lang: str = typer.Option(..., "--src-lang", "-s", help="源语言ISO 639-1 语种代码"),
    tgt_lang: str = typer.Option(..., "--tgt-lang", "-t", help="目标语言ISO 639-1 语种代码"),
    output_prefix: str = typer.Option("final", "--output-prefix", help="输出文件的前缀"),
    train_alignment: bool = typer.Option(False, "--train-alignment", help="是否训练词对齐模型"),
    apply_alignment_filter: bool = typer.Option(False, "--apply-alignment-filter", help="是否应用词对齐模型过滤"),
    alignment_filter_ratio: float = typer.Option(0.2, "--alignment-filter-ratio", help="词对齐模型过滤比例"),
):
    """
    使用 [OpusFilter](https://github.com/Helsinki-NLP/OpusFilter) 来清洗语料 
    """
    # 创建临时文件夹
    tmp_folder = tempfile.TemporaryDirectory(prefix="lanmttrainer")

    try:
        src_unit = UNIT_MAPPING[src_lang]
        tgt_unit = UNIT_MAPPING[tgt_lang]
        src_char = CHARACTER_MAPPING[src_lang]
        tgt_char = CHARACTER_MAPPING[tgt_lang]
        src_tokenizer = TOKENIZER_MAPPING[src_lang]
        tgt_tokenizer = TOKENIZER_MAPPING[tgt_lang]
    except KeyError:
        typer.echo(f"{src_lang} or {tgt_lang} is not supported",
                   err=True, fg="red")
        typer.Exit(1)

    rules_config = RULES_CONFIG.format(
        folder=folder,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        src_unit=src_unit,
        tgt_unit=tgt_unit,
        src_char=src_char,
        tgt_char=tgt_char,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    )

    configuration = yaml.loads(rules_config, Loader=yaml.Loader)
    of = OpusFilter(configuration)

    for step in range(1, 4):
        of.execute_step(step, overwrite=False)

    alignment_config = ALINGMENT_CONFIG.format(
        folder=folder,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        align_filter_remain=
    )

    if train_alignment:
        # 计算规则过滤后语料行数
        line_num = sum(1 for line in open(
            os.path.join(folder, f"{src_lang}.rules")))
        of.execute_step(4, overwrite=False)

    if apply_alignment_filter:
        of.execute_step(5, overwrite=False)
        of.execute_step(6, overwrite=False)


    # 关闭临时文件夹
    shutil.rmtree(tmp_folder.name)

if __name__ == "__main__":
    typer.run(opus_clean)
