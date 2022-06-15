"""Shared Large datasets into parts."""
import math
from pathlib import Path

import typer

from lanmttrainer.utils import count_lines


def shared_large_datasets(
    data_dir: str = typer.Argument(..., help="存放数据集的文件夹"),
    lang_paris: str = typer.Option(
        ..., "--lang-pairs", help="语种列表，使用`,`分隔，如`en-de,en-ru`等。"
    ),
    epoch_sents: int = typer.Option(..., "--epoch_sents", help="每个epoch最多包含的句对数。"),
    trainpref: str = typer.Option(
        "train",
        "--trainpref",
        help="数据文件前缀。如该参数指定为train，则语言对`en-de`的训练预料路径应为`data-dir`目录下 `train.en-de.en`, `train.en-de.de`。",
    ),
    suffix_pair: bool = typer.Option(
            False, "--suffix-pair", help="trainpref与语种标识之间是否加上语言对"
    )
):
    """Sharding very large datasets into parts."""
    # Wrap the data directory in Path object
    data_dir = Path(data_dir)

    # Split the language pairs
    lang_pairs = lang_paris.split(",")

    # Count the number of lines in each language pair
    if suffix_pair:
        pair2num = {
            pair: count_lines(data_dir / f"{trainpref}.{pair}.{pair.split('-')[0]}")
            for pair in lang_pairs
        }
    else:
        # TODO 这里判断文件不存在没有进行警告
        pair2num = {
            pair: count_lines(data_dir / f"{trainpref}.{pair.split('-')[0]}")
            for pair in lang_pairs
        }

    typer.echo(f"Num lines in each language pair: {pair2num}")

    sum_lines = sum(pair2num.values())
    num_epoch = math.ceil(sum_lines / epoch_sents)

    typer.echo(f"Total chunks: {num_epoch}")

    for pair, num in pair2num.items():
        assert num > 0, f"{pair} has no data"
        src_lang, tgt_lang = pair.split("-")
        fullprefix = trainpref + "." + pair if suffix_pair else trainpref
        chunck_size = math.ceil(num / num_epoch)

        srcfile = data_dir / f"{fullprefix}.{src_lang}"
        fout = None
        nthpart = 0
        for i, line in enumerate(srcfile.open()):
            if i % chunck_size == 0:
                if fout is not None:
                    fout.close()
                outfile = data_dir / f"part{nthpart}.{fullprefix}.{src_lang}"
                fout = open(outfile, "w")
                nthpart += 1
            fout.write(line)
        fout.close()

        tgtfile = data_dir / f"{fullprefix}.{tgt_lang}"
        fout = None
        nthpart = 0
        for i, line in enumerate(tgtfile.open()):
            if i % chunck_size == 0:
                if fout is not None:
                    fout.close()
                outfile = data_dir / f"part{nthpart}.{fullprefix}.{tgt_lang}"
                fout = open(outfile, "w")
                nthpart += 1
            fout.write(line)
        fout.close()


if __name__ == "__main__":
    typer.run(shared_large_datasets)
