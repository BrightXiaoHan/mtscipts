"""Script to train chinese+(another language) sentencepiece models."""
import io
from typing import List

import sentencepiece as spm
import typer


def train_spm_model(
    corpus_files: List[str] = typer.Option(
        ...,
        "--corpus",
        "-c",
        help="Corpus files to train spm models. Please ensure this parameter is the same order as parameter `languages`."
        "Meanwhile, please ensure the files are shuffled.",
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
):
    """Train spm models for source and target languages."""
    langs = set(languages)
    model = io.BytesIO()
    sentence_iterator = get_sentence_iterator(
        corpus_files, languages, input_sentence_size, sample_temprature
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
