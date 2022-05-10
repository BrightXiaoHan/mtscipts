import typer
import os
import io
import enum
import fileinput
import subprocess
from typing import List, Optional
import sentencepiece as spm


class OptionalModes(str, enum.Enum):
    join_bpe_common_chars = "join_bpe_common_chars"


def train_spm_model(
    corpus_files: List[str] = typer.Argument(..., help="Corpus files to train spm models."),
    output_folder_with_prefix: str = typer.Option(".", help="Output folder with prefix. For example: /tmp/enzh which will create /tmp/enzh.model and /tmp/enzh.vocab"),
    src_lang: Optional[str] = typer.Option(None, help="ISO 639-1 language code for source language"),
    tgt_lang: Optional[str] = typer.Option(None, help="ISO 639-1 language code for target language"),
    mode: OptionalModes = typer.Option(OptionalModes.join_bpe_common_chars, help="Mode to use for training spm models."),
):
    """
    Train spm models for source and target languages.
    """
    if mode == OptionalModes.join_bpe_common_chars:
        pwd = os.path.dirname(os.path.abspath(__file__))
        common_symbols = ",".join(map(str.strip, fileinput.input([
            os.path.join(pwd, "assets", "common_zh_chars.txt"),
            os.path.join(pwd, "assets", "domain_symbols.txt"),
            os.path.join(pwd, "assets", "language_prefix_symbols.txt"),
            os.path.join(pwd, "assets", "term_mask_symbols.txt")
        ])))
        model = io.BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=fileinput.input(corpus_files),
            model_writer=model,
            vocab_size=32000,
            accept_language=",".join([src_lang, tgt_lang]) if src_lang and tgt_lang else "",
            input_sentence_size=20000000,
            character_coverage=0.99999,
            model_type="bpe",
            max_sentencepiece_length=6,
            user_defined_symbols=common_symbols,
            shuffle_input_sentence=True,
            unk_surface=""
        )
        with open(output_folder_with_prefix + ".model", 'wb') as f:
            f.write(model.getvalue())

        subprocess.run(
            f"spm_export_vocab --model {output_folder_with_prefix+'.model'} --output {output_folder_with_prefix+'.vocab'}",
            shell=True
        )
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")


if __name__ == "__main__":
    typer.run(train_spm_model)
