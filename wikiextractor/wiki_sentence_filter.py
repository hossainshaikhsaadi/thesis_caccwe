import sys
from pathlib import Path

from blingfire import text_to_sentences, text_to_words


def main():
    wiki_dump_file_in = Path(sys.argv[1])
    wiki_dump_file_out = wiki_dump_file_in.parent / \
        f'{wiki_dump_file_in.stem}_filtered{wiki_dump_file_in.suffix}'

    print(f'Pre-processing {wiki_dump_file_in} to {wiki_dump_file_out}...')
    with open(wiki_dump_file_out, 'w', encoding='utf-8') as out_f:
        with open(wiki_dump_file_in, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                words = line.strip().split()
                if len(words) >= 5:
                    out_f.write(line.strip() + '\n')
    print(f'Successfully filtered {wiki_dump_file_in} to {wiki_dump_file_out}...')


if __name__ == '__main__':
    main()