# Author: Tzu-Quan Lin
# Reference: https://github.com/espnet/espnet/blob/master/egs2/ml_superb/asr1/local/data_prep.py

import argparse
import os
import re
import string

DATA = [
    "ALFFA",
    "LAD",
    "M-AILABS",
    "NST",
    "commonvoice",
    "fleurs",
    "googlei18n_asr",
    "googlei18n_tts",
    "mls",
    "nchlt",
    "swc",
    "voxforge",
    "mexico-el",
    "voxpopuli",
]

RESERVE_LANG = [
    "dan",
    "lit",
    "tur",
    "srp",
    "vie",
    "kaz",
    "zul",
    "tsn",
    "epo",
    "frr",
    "tok",
    "umb",
    "bos",
    "ful",
    "ceb",
    "luo",
    "kea",
    "sun",
    "tso",
    "tos",
]

FEW_SHOT_SELECTED_DATA = {
    "lit": [
        "cv_lit_000026",
        "cv_lit_000105",
        "fleurs_lit_000097",
        "mls_lit_000040",
        "voxpopuli_lit_001707",
    ],
    "dan": [
        "NST_dan_000007",
        "NST_dan_000072",
        "cv_dan_000158",
        "fleurs_dan_000058",
        "fleurs_dan_000071",
    ],
    "tur": [
        "cv_tur_000095",
        "cv_tur_000101",
        "fleurs_tur_000060",
        "fleurs_tur_000069",
        "fleurs_tur_000095",
    ],
    "srp": [
        "cv_srp_000078",
        "cv_srp_000145",
        "fleurs_srp_000068",
        "fleurs_srp_000093",
        "fleurs_srp_000105",
    ],
    "vie": [
        "cv_vie_000105",
        "cv_vie_000128",
        "fleurs_vie_000067",
        "fleurs_vie_000068",
        "fleurs_vie_000077",
    ],
    "kaz": [
        "cv_kaz_000080",
        "cv_kaz_000097",
        "cv_kaz_000111",
        "fleurs_kaz_000036",
        "fleurs_kaz_000066",
    ],
    "zul": [
        "fleurs_zul_000049",
        "fleurs_zul_000057",
        "nchlt_zul_000027",
        "nchlt_zul_000090",
        "nchlt_zul_000104",
    ],
    "tsn": [
        "googlei18n-tts_tsn_000026",
        "googlei18n-tts_tsn_000044",
        "googlei18n-tts_tsn_000108",
        "nchlt_tsn_000001",
        "nchlt_tsn_000032",
    ],
    "epo": [
        "cv_epo_000006",
        "cv_epo_000039",
        "cv_epo_000063",
        "cv_epo_000066",
        "cv_epo_000076",
    ],
    "frr": [
        "cv_frr_000023",
        "cv_frr_000086",
        "cv_frr_000095",
        "cv_frr_000102",
        "cv_frr_000104",
    ],
    "tok": [
        "cv_tok_000004",
        "cv_tok_000011",
        "cv_tok_000030",
        "cv_tok_000084",
        "cv_tok_000101",
    ],
    "umb": [
        "fleurs_umb_000028",
        "fleurs_umb_000029",
        "fleurs_umb_000033",
        "fleurs_umb_000040",
        "fleurs_umb_000047",
    ],
    "bos": [
        "fleurs_bos_000067",
        "fleurs_bos_000078",
        "fleurs_bos_000080",
        "fleurs_bos_000088",
        "fleurs_bos_000090",
    ],
    "ful": [
        "fleurs_ful_000055",
        "fleurs_ful_000059",
        "fleurs_ful_000067",
        "fleurs_ful_000076",
        "fleurs_ful_000081",
    ],
    "ceb": [
        "fleurs_ceb_000054",
        "fleurs_ceb_000064",
        "fleurs_ceb_000069",
        "fleurs_ceb_000071",
        "fleurs_ceb_000080",
    ],
    "luo": [
        "fleurs_luo_000056",
        "fleurs_luo_000062",
        "fleurs_luo_000067",
        "fleurs_luo_000073",
        "fleurs_luo_000077",
    ],
    "kea": [
        "fleurs_kea_000052",
        "fleurs_kea_000070",
        "fleurs_kea_000078",
        "fleurs_kea_000083",
        "fleurs_kea_000085",
    ],
    "sun": [
        "googlei18n-asr_sun_000001",
        "googlei18n-asr_sun_000007",
        "googlei18n-asr_sun_000041",
        "googlei18n-asr_sun_000099",
        "googlei18n-asr_sun_000106",
    ],
    "tso": [
        "nchlt_tso_000035",
        "nchlt_tso_000040",
        "nchlt_tso_000089",
        "nchlt_tso_000104",
        "nchlt_tso_000125",
    ],
    "tos": [
        "mexico-el_tos_000006",
        "mexico-el_tos_000122",
        "mexico-el_tos_000152",
        "mexico-el_tos_000496",
        "mexico-el_tos_000563",
    ],
}

def prep_data(args):
    DIR = args.data_dir 
    OUT = args.out_dir
    DURATION = args.duration
    
    os.makedirs(OUT, exist_ok=True)
    train_fp = open(os.path.join(OUT,"train.csv"), "w")
    dev_fp = open(os.path.join(OUT,"dev.csv"), "w")
    test_fp = open(os.path.join(OUT,"test.csv"), "w")
    idx_fp = open(os.path.join(OUT,"lang2idx.txt"), "w")

    langs_info = {}

    # iterate through dataset
    for dataset in DATA:
        langs = os.listdir(os.path.join(DIR, dataset))
        for lang in langs:
            reserve_flag = False
            if lang in RESERVE_LANG:
                reserve_flag = True  # skip reserve lange for zero-shot
            if lang not in langs_info:
                langs_info[lang] = []
            langs_info[lang].append(dataset)

            # process train
            train_transcript = open(
                os.path.join(
                    DIR,
                    dataset,
                    lang,
                    "transcript_{}_train.txt".format(DURATION),
                ),
                "r",
                encoding="utf-8",
            )
            for line in train_transcript.readlines():
                line = line.strip().split(maxsplit=2)
                utt_id, _, _ = line
                if reserve_flag and utt_id not in FEW_SHOT_SELECTED_DATA[lang]:
                    continue
                uttr_pth = os.path.join(
                    DIR,
                    dataset,
                    lang,
                    "wav/{}.wav".format(utt_id)
                )
                train_fp.write(f"{lang},{uttr_pth}\n")
               
            train_transcript.close()

            # process dev
            dev_transcript = open(
                os.path.join(DIR, dataset, lang, "transcript_10min_dev.txt"),
                "r",
                encoding="utf-8",
            )
            for line in dev_transcript.readlines():
                line = line.strip().split(maxsplit=2)
                utt_id, _, _ = line
                uttr_pth = os.path.join(
                    DIR,
                    dataset,
                    lang,
                    "wav/{}.wav".format(utt_id)
                )
                dev_fp.write(f"{lang},{uttr_pth}\n")
               
            dev_transcript.close()

            # process test
            test_transcript = open(
                os.path.join(DIR, dataset, lang, "transcript_10min_test.txt"),
                "r",
                encoding="utf-8",
            )
            for line in test_transcript.readlines():
                line = line.strip().split(maxsplit=2)
                utt_id, _, _ = line
                uttr_pth = os.path.join(
                    DIR,
                    dataset,
                    lang,
                    "wav/{}.wav".format(utt_id)
                )
                test_fp.write(f"{lang},{uttr_pth}\n")
    
            test_transcript.close()

    for i, lang in enumerate(langs_info.keys()):
        idx_fp.write(f"{i},{lang}\n")

    print("{} languages processed, ".format(len(langs_info)))
    for lang in langs_info.keys():
        dataset = " ".join(langs_info[lang])
        print(f"{lang} {len(langs_info[lang])} {dataset}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--duration", type=str, default="1h")
    args = parser.parse_args()
    prep_data(args)
