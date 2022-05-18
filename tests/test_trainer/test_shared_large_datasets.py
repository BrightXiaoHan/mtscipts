import glob
import os
import random
from lanmttrainer.trainer.fairseq.shared_large_datasets import \
    shared_large_datasets
from lanmttrainer.utils import count_lines


def test_shared_large_datasets(tmpdir):
    lang_paris = ["en-zh", "en-fr", "en-de", "en-es", "en-ru", "en-it"]
    pair2num = {pair: random.randint(100, 1000) for pair in lang_paris}
    trainprefix = "train"

    for pair in lang_paris:
        src, tgt = pair.split("-")
        srcfile = tmpdir.join(f"{trainprefix}.{pair}.{src}")
        tgtfile = tmpdir.join(f"{trainprefix}.{pair}.{tgt}")

        srcfile.write("\n".join(["a" for _ in range(pair2num[pair])]))
        tgtfile.write("\n".join(["b" for _ in range(pair2num[pair])]))

    shared_large_datasets(tmpdir, ",".join(lang_paris), 300, trainprefix)

    for pair in lang_paris:
        src, tgt = pair.split("-")
        src_path = os.path.join(tmpdir.strpath, f"part*.{trainprefix}.{pair}.{src}")
        tgt_path = os.path.join(tmpdir.strpath, f"part*.{trainprefix}.{pair}.{tgt}")

        src_files = glob.glob(src_path)
        tgt_files = glob.glob(tgt_path)

        src_all_lines = sum([count_lines(f) for f in src_files])
        tgt_all_lines = sum([count_lines(f) for f in tgt_files])

        assert src_all_lines == tgt_all_lines == pair2num[pair]
