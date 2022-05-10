"""
Add domain tag to the start of each sentence in the input file.
"""
import typer
import fileinput


def add_domain_tags(
    input_file: str = typer.Argument(..., help="Input file to add domain tags."),
    domain_tag: str = typer.Argument(..., help="Domain tag to add to the start of each sentence."),
    output_suffix: str = typer.Option(
        "with_domain_tags", help="Output file suffix. For example: if input file is 'input.txt' and output_suffix is 'with_domain_tags', then output file will be 'input.txt.with_domain_tags'.")
):
    """
    Add domain tag to the start of each sentence in the input file.
    """
    with open(f"{input_file}.{output_suffix}", 'w', encoding='utf-8') as f:
        for line in fileinput.input([input_file]):
            print(f"｟{domain_tag}｠ {line}".strip(), file=f)


if __name__ == "__main__":
    typer.run(add_domain_tags)
