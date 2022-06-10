"""Script to train multilingual sentencepiece models."""
import io
from collections import defaultdict
from typing import List

import sentencepiece as spm
import typer
from more_itertools import interleave, take

from lanmttrainer.utils import count_lines


def get_sentence_iterator(
    corpus_files, languages, sample_temprature
):
    """Build data iterator for sentencepiece trainer."""
    typer.echo("Counting length of each file.")
    lengths = [count_lines(file_path) for file_path in corpus_files]
    total_length = sum(lengths)

    typer.echo("Counting length of each file done.")
    lang2length = defaultdict(int)

    # count total length of each language
    for lang, length in zip(languages, lengths):
        lang2length[lang] += length

    # collection files for each language
    lang2files = defaultdict(list)
    for lang, file_path in zip(languages, corpus_files):
        lang2files[lang].append(file_path)

    sum_length = sum(lengths)
    lang2temp = {
        lang: pow(length / sum_length, 1.0 / sample_temprature)
        for lang, length in lang2length.items()
    }
    sum_temp = sum(lang2temp.values())
    lang2sample = {
        lang: int(total_length * temp / sum_temp)
        for lang, temp in lang2temp.items()
    }

    typer.echo("Sample size of each language: {}".format(lang2sample))

    for lang, files in lang2files.items():
        for line in take(
            lang2sample[lang],
            interleave(*[open(file_path, "r") for file_path in files]),
        ):
            yield line.strip()


def train_spm_model(
    corpus_files: List[str] = typer.Option(
        ...,
        "--corpus",
        "-c",
        help="Corpus files to train spm models. Please ensure this parameter is the same order as parameter `languages`."
    ),
    vocab_size: int = typer.Option(
        ..., "--vocab-size", "-s", help="Vocabulary size to train spm models."
    ),
    input_sentence_size: int = typer.Option(
        10000000,
        "--input-sentence-size",
        "-i",
        help="Input sentence size to train spm models.",
    ),
    languages: List[str] = typer.Option(
        ..., "--langs", "-l", help="Languanges ISO 639-1 codes of each files."
    ),
    output_folder_with_prefix: str = typer.Option(
        "spm",
        "--output-folder-with-prefix",
        "-o",
        help="Output folder with prefix. For example: /tmp/enzh which will create /tmp/enzh.model.",
    ),
    sample_temprature: float = typer.Option(
        1,
        "--sample-temprature",
        "-t",
        help="Sample temprature of each language. Useful for training multilingual models.",
    ),
):
    """Train spm models for source and target languages."""
    langs = set(languages)
    model = io.BytesIO()
    sentence_iterator = get_sentence_iterator(
        corpus_files, languages, sample_temprature
    )

    spm.SentencePieceTrainer.train(
        sentence_iterator=sentence_iterator,
        model_writer=model,
        vocab_size=vocab_size,
        accept_language=",".join(langs),
        input_sentence_size=input_sentence_size,
        character_coverage=1,
        model_type="unigram",
        shuffle_input_sentence=True,
        unk_surface="",
    )
    with open(output_folder_with_prefix + ".model", "wb") as f:
        f.write(model.getvalue())


if __name__ == "__main__":
    typer.run(train_spm_model)
