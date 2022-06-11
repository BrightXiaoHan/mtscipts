"""Apply subword regularization to given text file."""
import fileinput

import sentencepiece as spm
import typer


def spm_encode_with_subword_regularization(
    model: str = typer.Option(
        ..., "--model", "-m", help="Path to sentencepiece model."
    ),
    input_file: str = typer.Option(
        ..., "--input-file", "-i", help="Path to input file."
    ),
    output_suffix: str = typer.Option(
        "subr", "--output-suffix", "-o", help="Suffix to add to output file."
    ),
    sample_size: int = typer.Option(
        5, "--sample-size", "-s", help="Number of samples to generate."
    ),
    alpha: float = typer.Option(
        0.1, "--alpha", "-a", help="Alpha value for subword regularization."
    ),
):
    """Apply subword regularization to given text file."""
    s = spm.SentencePieceProcessor(model_file=model)

    outf = open(f"{input_file}.{output_suffix}", "w")
    for line in fileinput.input(input_file):
        line = line.strip()
        pieces = s.encode(line, out_type=str, enable_sampling=False)
        print(" ".join(pieces), file=outf)
        for _ in range(sample_size - 1):
            pieces = s.encode(
                line, out_type=str, enable_sampling=True, alpha=alpha, nbest_size=-1
            )
            print(" ".join(pieces), file=outf)

    outf.close()


if __name__ == "__main__":
    typer.run(spm_encode_with_subword_regularization)
