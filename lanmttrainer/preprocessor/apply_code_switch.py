"""Apply code switch to the corpus."""
import os

import typer
import WordAlignFilters.TermReplace as tmr


def apply_code_switch(
    folder: str = typer.Argument(..., help="存放语料数据的文件夹"),
    translation_table: str = typer.Option(
        ..., "--translation-table", "-t", help="翻译表位置"
    ),
    alignment_table: str = typer.Option(..., "--alignment-table", "-a", help="对齐表位置"),
):
    """Apply code switch to the corpus."""
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if len(files) != 2 or "zh" not in files:
        typer.echo(
            f"{folder} 不是一个合法的语料文件夹，请检查后重试。必须包含两个文件，以语种简称命名：如`zh, en`",
            err=True,
            color="red",
        )
        typer.Exit(1)

    zh_file, foreign_file = files

    for output_path, direction in zip(
        [
            os.path.join(folder, f"{zh_file}-{foreign_file}"),
            os.path.join(folder, f"{foreign_file}-{zh_file}"),
        ],
        [True, False],
    ):
        os.makedirs(output_path, exist_ok=True)
        tmr.run(
            foreign_file,
            translation_table,
            alignment_table,
            thread=0,  # 阈值
            zh_files=[os.path.join(folder, zh_file)],
            foreign_files=[os.path.join(folder, foreign_file)],
            output_path=output_path,
            z2f=direction,
        )

    # 合并并乱序文件


if __name__ == "__main__":
    typer.run(apply_code_switch)
