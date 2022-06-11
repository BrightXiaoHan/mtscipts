import argparse
import json
import os
import random
from collections import defaultdict

import regex as re
from nltk import IBMModel
from sacremoses import MosesDetokenizer
from tqdm import tqdm
from tqdm.contrib import tzip
from yaml import FullLoader, load

abs_path = os.path.abspath(os.path.dirname(__file__))


class Align:
    def __init__(
        self,
        foreign,
        translation_table,
        alignment_table,
        z2f,
        max_num=10,
        thread=0.001,
        blank=False,
    ):
        self.translation_table = defaultdict(
            lambda: defaultdict(lambda: IBMModel.MIN_PROB)
        )
        self.alignment_table = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: IBMModel.MIN_PROB))
            )
        )
        with open(translation_table, encoding="utf-8") as tr, open(
            alignment_table, encoding="utf-8"
        ) as al:
            translation_dict = json.load(tr)
            for src in tqdm(
                translation_dict, desc="loading translation table to defaultdict class"
            ):
                pro_table = translation_dict.get(src)
                for tgt in pro_table:
                    pro = pro_table.get(tgt)
                    self.translation_table[src][tgt] = pro
            del translation_dict
            alignment_dict = json.load(al)
            for j in tqdm(
                alignment_dict, desc="loading alignment table to defaultdict class"
            ):
                i_table = alignment_dict.get(j)
                for i in i_table:
                    m_table = i_table.get(i)
                    for m in m_table:
                        n_table = m_table.get(m)
                        for n in n_table:
                            pro = n_table.get(n)
                            self.alignment_table[j][i][m][n] = pro
            del alignment_dict
        self.md = MosesDetokenizer(lang=foreign)
        self.zh = None
        self.foreign = None
        self.m = None
        self.l = None
        self.alignment = None
        self.re_punc = re.compile("[\p{P}]")
        self.thread = thread
        self.zh_str = None
        self.foreign_str = None
        self.z2f = z2f
        self.max_num = max_num
        self.blank = blank

    def get_align(self, zh, foreign, params, merged_line):
        """
        对齐函数
        :param zh: 中文句子
        :param foreign: 英文句子
        :return:
        """
        self.zh_str = zh
        self.foreign_str = foreign
        zh_line, foreign_line = merged_line.split("|||")
        self.zh = zh_line.strip().split()
        self.foreign = foreign_line.strip().split()
        self.m = len(self.zh)
        self.l = len(self.foreign)
        self.alignment = params

    def get_probability(self):
        """
        获取词对齐概率表
        :return:
        """
        pro_items = {}
        flag = False
        foreign_ind = -1
        merge_list = ["", None, 0, 0, -1, -1, -1]
        start_ind = 0
        for aligns in self.alignment.split():
            pre_foreign_ind = foreign_ind
            zh_ind, foreign_ind = aligns.split("-")
            zh_ind, foreign_ind = int(zh_ind), int(foreign_ind)
            t = self.zh[zh_ind]
            s = self.foreign[foreign_ind]
            pro = (
                self.translation_table[t][s]
                * self.alignment_table[str(foreign_ind + 1)][str(zh_ind + 1)][
                    str(self.l)
                ][str(self.m)]
            )
            if foreign_ind == pre_foreign_ind:
                end_ind = zh_ind
                if not flag:
                    flag = True
            else:
                start_ind = zh_ind
                end_ind = zh_ind
                if flag:
                    pro_items[
                        (
                            merge_list[0],
                            merge_list[1],
                            merge_list[4],
                            merge_list[5],
                            merge_list[6],
                        )
                    ] = (merge_list[2] / merge_list[3])
                    merge_list = ["", None, 0, 0, -1, -1, -1]
                    flag = False
                else:
                    if merge_list[1]:
                        pro_items[
                            (
                                merge_list[0],
                                merge_list[1],
                                merge_list[4],
                                merge_list[5],
                                merge_list[6],
                            )
                        ] = (merge_list[2] / merge_list[3])
                        merge_list = ["", None, 0, 0, -1, -1, -1]
            merge_list[0] += t
            merge_list[1] = s
            merge_list[2] += pro
            merge_list[3] += 1
            merge_list[4] = start_ind
            merge_list[5] = end_ind
            merge_list[6] = foreign_ind
        if merge_list[3]:
            pro_items[
                (
                    merge_list[0],
                    merge_list[1],
                    merge_list[4],
                    merge_list[5],
                    merge_list[6],
                )
            ] = (merge_list[2] / merge_list[3])
        return pro_items

    def term_replace(self):
        """
        获取术语保护后的原文译文
        :return:
        """
        pro_items = self.get_probability().items()
        selected_items = [item for item in pro_items if item[1] > self.thread]
        random.shuffle(selected_items)
        max_nums = min(len(pro_items) // 5 + 1, self.max_num)
        max_nums = random.randint(1, max(int(max_nums), 1))
        count = 0
        zh = self.zh.copy()
        foreign = self.foreign.copy()
        symbol_labels = []
        for item in selected_items:
            if not self.re_punc.search(item[0][0] + item[0][1]):
                if abs(item[0][2] - item[0][3]) >= 2:
                    continue
                zh_symbol = item[0][0]
                foreign_symbol = item[0][1]
                symbol_labels.append((zh_symbol, foreign_symbol))
                if self.z2f:
                    if self.blank:
                        zh[item[0][2]] = " " + foreign_symbol + " "
                    else:
                        zh[item[0][2]] = foreign_symbol
                    for i in range(item[0][2] + 1, item[0][3] + 1):
                        zh[i] = None
                else:
                    foreign[item[0][4]] = zh_symbol
                count += 1
            if count == max_nums:
                break
        if self.z2f:
            if self.blank:
                term_zh = re.sub(r"\s+", " ", "".join(i for i in zh if i))
            else:
                term_zh = " ".join(i for i in zh if i)
                processor = self.get_remove_whitespace_postprocessor()
                term_zh = processor(term_zh)
            term_foreign = self.foreign_str
        else:
            term_foreign = self.md.detokenize(foreign)
            term_zh = self.zh_str
        return term_zh, term_foreign, symbol_labels

    @staticmethod
    def get_remove_whitespace_postprocessor(_=None):
        """
        去除文本中的所有空格，中文按字分开时可以将字符间空格去掉。
        >>> processor = get_remove_whitespace_postprocessor()
        >>> text = "本 • 富 兰 克 林 （ Ben Franklin ） 称 德 国 人 愚 蠢 而 剽 悍 。"
        >>> processor(text)
        '本•富兰克林（ Ben Franklin ）称德国人愚蠢而剽悍。'
        """
        re_han = re.compile("([^A-Za-z])(\\s+)")
        re_han2 = re.compile("(\\s+)([^A-Za-z])")

        def postprocessor(line):
            while re_han.search(line):
                line = re_han.sub("\\g<1>", line)

            while re_han2.search(line):
                line = re_han2.sub("\\g<2>", line)
            return line

        return postprocessor


def run(
    foreign,
    translation_table,
    alignment_table,
    thread,
    zh_files,
    foreign_files,
    aligned_file,
    merged_file,
    output_path,
    z2f,
    max_num=10,
    blank=False,
):
    align = Align(
        foreign, translation_table, alignment_table, z2f, max_num, thread, blank
    )
    if z2f:
        t = "z2f"
    else:
        t = "f2z"
    for zh_file, foreign_file in zip(zh_files, foreign_files):
        with open(zh_file, encoding="utf-8") as zf, open(
            foreign_file, encoding="utf-8"
        ) as ff, open(aligned_file) as af, open(merged_file) as mf, open(
            f"{output_path}/{t}.zh.term", "w", encoding="utf-8"
        ) as ztf, open(
            f"{output_path}/{t}.{foreign}.term", "w", encoding="utf-8"
        ) as ftf, open(
            f"{output_path}/symbol.labels", "w", encoding="utf-8"
        ) as slf:
            for zh, foreign, aligned_params, merged_line in tzip(
                zf.readlines(), ff.readlines(), af.readlines(), mf.readlines()
            ):
                if merged_line == "<error> ||| <error>\n":
                    term_zh, term_foreign, symbol_labels = (
                        zh.strip(),
                        foreign.strip(),
                        [],
                    )
                else:
                    align.get_align(
                        zh.strip(),
                        foreign.strip(),
                        aligned_params.strip(),
                        merged_line.strip(),
                    )
                    term_zh, term_foreign, symbol_labels = align.term_replace()
                ztf.write(term_zh + "\n")
                ftf.write(term_foreign + "\n")
                slf.write(str(symbol_labels) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Term replace script")
    parser.add_argument(
        "--config", type=str, default=False, help="Filter follow with config file"
    )
    parser.add_argument("-fl", type=str, help="forein language")
    parser.add_argument("-ff", nargs="+", help="foreign file")
    parser.add_argument("-zf", nargs="+", help="Chinese file")
    parser.add_argument(
        "--thread", type=float, default=0.001, help="Term replace thread, default 0.001"
    )
    parser.add_argument("-o", type=str, default=".", help="Output folder")
    parser.add_argument("-z2f", type=bool, help="Translation direction")
    parser.add_argument(
        "--max_num", type=int, default=10, help="Max number of term replace"
    )
    parser.add_argument(
        "--blank",
        type=bool,
        default=False,
        help="Whether to add blank between Chinese and foreign",
    )
    parser.add_argument("--aligned_file", type=str, required=True, help="Aligned file")
    parser.add_argument("--merged_file", type=str, required=True, help="Merged file")
    args = parser.parse_args()
    if args.config:
        datas = load(open(args.config, encoding="utf-8"), Loader=FullLoader)
        foreign = datas["foreign_language"]
        output_path = datas.get("output_path", ".")
        thread = datas.get("thread", 0.001)
        zh_files = datas["zh_files"]
        foreign_files = datas["foreign_files"]
        z2f = datas["z2f"]
        max_num = datas.get("max_num", 10)
        blank = datas.get("blank", False)
        aligned_file = datas["aligned_file"]
        merged_file = datas["merged_file"]
    else:
        output_path = args.o
        thread = args.thread
        foreign = args.fl
        zh_files = args.zf
        foreign_files = args.ff
        z2f = args.z2f
        max_num = args.max_num
        blank = args.blank
        aligned_file = args.aligned_file
        merged_file = args.merged_file
    translation_table = f"{abs_path}/model_{foreign}/translation.table"
    alignment_table = f"{abs_path}/model_{foreign}/alignment.table"
    run(
        foreign,
        translation_table,
        alignment_table,
        thread,
        zh_files,
        foreign_files,
        aligned_file,
        merged_file,
        output_path,
        z2f,
        max_num,
        blank,
    )


if __name__ == "__main__":
    main()
