import fileinput
from pathlib import Path
from typing import Union

import typer
from more_itertools import divide

from lanmttrainer.utils import count_lines


def shared_large_datasets(
        data_dir: Union[Path, str] = typer.Argument(..., help="存放数据集的文件夹"),
        lang_paris: str = typer.Option(..., "--lang-pairs", help="语种列表，使用`,`分隔，如`en-de,en-ru`等。"),
        epoch_sents: int = typer.Option(..., "--epoch_sents", help="每个epoch最多包含的句对数。"),
        trainprefix: str = typer.Option(
            "train",
            "--trainprefix",
            help="数据文件前缀。如该参数指定为train，则语言对`en-de`的训练预料路径应为`data-dir`目录下 `train.en-de.en`, `train.en-de.de`。",
        ),
):
    """Sharding very large datasets into parts."""
    # Wrap the data directory in Path object
    data_dir = Path(data_dir)

    # Split the language pairs
    lang_pairs = lang_paris.split(",")

    # Count the number of lines in each language pair
    pair2num = {
        pair: count_lines(
            data_dir / f"{trainprefix}.{pair}.{pair.split('-')[0]}"
        )
        for pair in lang_pairs
    }

    sum_lines = sum(pair2num.values())
    num_epoch = sum_lines // epoch_sents + 1

    for pair in lang_pairs:
        src_lang, tgt_lang = pair.split("-")

        srcfile = data_dir / f"{trainprefix}.{pair}.{src_lang}"

        for i, chunk in enumerate(divide(num_epoch, srcfile.open())):
            outfile = data_dir / f"part{i}.{trainprefix}.{pair}.{src_lang}"
            with open(outfile, "w") as fout:
                for line in chunk:
                    fout.write(line)
        tgtfile = data_dir / f"{trainprefix}.{pair}.{tgt_lang}"

        for i, chunk in enumerate(divide(num_epoch, tgtfile.open())):
            outfile = data_dir / f"part{i}.{trainprefix}.{pair}.{tgt_lang}"
            with open(outfile, "w") as fout:
                for line in chunk:
                    fout.write(line)
