import os.path
from os import path
from collections import OrderedDict
import json

train = dict(
    abs="C:\\Users\\ganes\\PycharmProjects\\transformers\\examples\\token-classification\\data\\train\\chemprot_training_abstracts.tsv",
    ent="C:\\Users\\ganes\\PycharmProjects\\transformers\\examples\\token-classification\\data\\train\\chemprot_training_entities.tsv",
    dest="C:\\Users\\ganes\\PycharmProjects\\transformers\\examples\\token-classification\\pre_tagged\\train.txt.tmp")

test = dict(
    abs="C:\\Users\\ganes\\PycharmProjects\\transformers\\examples\\token-classification\\data\\test\\chemprot_test_abstracts_gs.tsv",
    ent="C:\\Users\\ganes\\PycharmProjects\\transformers\\examples\\token-classification\\data\\test\\chemprot_test_entities_gs.tsv",
    dest="C:\\Users\\ganes\\PycharmProjects\\transformers\\examples\\token-classification\\pre_tagged\\test.txt.tmp")

dev = dict(
    abs="C:\\Users\\ganes\\PycharmProjects\\transformers\\examples\\token-classification\\data\\dev\\chemprot_development_abstracts.tsv",
    ent="C:\\Users\\ganes\\PycharmProjects\\transformers\\examples\\token-classification\\data\\dev\\chemprot_development_entities.tsv",
    dest="C:\\Users\\ganes\\PycharmProjects\\transformers\\examples\\token-classification\\pre_tagged\\dev.txt.tmp")


def clean_white_space(sentence):
    import re
    sentence = sentence.strip()
    sentence = re.sub(r'(\s|\u180B|\u200B|\u200C|\u200D|\u2060|\uFEFF)+', ' ', sentence)
    return sentence.strip()


def check_file_exists(file_path):
    if not path.exists(file_path):
        print("Error File Missing.", file_path)
        raise ValueError("Error File Missing.", file_path)


def get_file_obj(file_path, mode="r+", encoding="utf8"):
    check_file_exists(file_path)
    if "b" in mode:
        return open(file_path, mode), file_path
    return open(file_path, mode, encoding=encoding), file_path


def generate_entity_map(all_rows, tsv_file_path=None):
    tags_set = set("O")
    entities_mapper = OrderedDict()
    index_based_tagging_res = OrderedDict()
    entities_indexer = []
    for row in all_rows:
        entity = row.replace("\n", " ").replace("\t", "$$$")
        entity = clean_white_space(entity)
        entity = entity.split("$$$")
        if len(entity) != 6:
            print("ERROR: Not enough Entity information! ", tsv_file_path)
            return

        entity_pmid = int(str(entity[0]).strip())
        entity_type = str(entity[2].strip())
        entity_start = int(str(entity[3]).strip())
        entity_end = int(str(entity[4]).strip())
        entity_val = str(entity[5])
        entities_dict = OrderedDict()
        entities_tag_list = []
        index_based_tagging = {}
        # entities_indexer.append([entity_val, (entity_start, entity_end), entity_type])

        entities_split = entity_val.split(" ")
        tag_prefix = "B-"
        flag = False
        for ele in entities_split:
            ele = clean_white_space(ele)
            if flag:
                tag_prefix = "I-"
            tags_set.add(tag_prefix + entity_type)
            entities_tag_list.append(
                [entity_start, ele, (entity_start, entity_start + len(ele)), tag_prefix + entity_type])
            for i in range(entity_start, entity_start + len(ele)):
                index_based_tagging[i] = tag_prefix + entity_type
            flag = True
            entity_start = entity_start + len(ele) + 1
        if entity_pmid in entities_mapper:
            entities_mapper[entity_pmid].extend(entities_tag_list)
            for ele in index_based_tagging:
                index_based_tagging_res[entity_pmid][ele] = index_based_tagging[ele]
            # index_based_tagging_res[entity_pmid].extend(index_based_tagging)
        else:
            entities_mapper[entity_pmid] = entities_tag_list
            index_based_tagging_res[entity_pmid] = index_based_tagging
    print("Tags:", tags_set)
    for line in entities_mapper:
        # print(entities_mapper[line])
        # exit(1)
        entities_mapper[line] = sorted(entities_mapper[line], key=lambda x: x[0])
    # print(index_based_tagging_res)
    # exit(1)
    return entities_mapper, index_based_tagging_res


def generate_abstract_map(all_rows, abs_file_path=None):
    abstract_mapper = OrderedDict()
    for abs_line in all_rows:
        abs_line = abs_line.replace("\n", " ").replace("\t", "$$$")
        abs_line = clean_white_space(abs_line)
        abs_split_t = abs_line.split("$$$")
        entity_pmid = int(abs_split_t[0].strip())
        name_content = " ".join(abs_split_t[1:])
        name_content = clean_white_space(name_content)
        abs_split_s = name_content.split(" ")
        some_res = OrderedDict()
        start = 0
        for ele in abs_split_s:
            # ele = clean_white_space(ele)
            if ele.strip() == "":
                start = start + 1
                continue
            some_res[start] = "O"
            start = start + len(ele) + 1
        abstract_mapper[entity_pmid] = (some_res, abs_split_s)
    return abstract_mapper


def gen_tags(data_obj):
    # for entities
    entities_file_obj = open(data_obj['ent'], "r+", encoding="utf8")
    entities_mapper_obj, index_based_tagging_res = generate_entity_map(entities_file_obj.readlines(), train['ent'])

    abstract_file_obj = open(data_obj['abs'], "r+", encoding="utf8")
    abstract_mapper_obj = generate_abstract_map(abstract_file_obj.readlines(), train['abs'])

    final_res = []
    final_csv = data_obj['dest']
    word_tag_obj = open(final_csv, "w", encoding='utf-8')

    print("Generating final tokens...")

    print("---------------------------------------------------------")

    # for pm_id in entities_mapper_obj:
    #     for data in entities_mapper_obj[pm_id]:
    #         abstract_mapper_obj[pm_id][0][data[0]] = data[1]

    for pm_id in abstract_mapper_obj:
        line = abstract_mapper_obj[pm_id]
        tags = line[0]
        words = line[1]
        # noinspection PyRedeclaration
        start_idx = 0
        for wd in words:
            tag = "O"
            # if start_idx in index_based_tagging_res:
            #     tag = index_based_tagging_res[start_idx]
            for i in range(start_idx, start_idx + len(wd)):
                index_based_tagging_res2 = index_based_tagging_res[pm_id]
                if i in index_based_tagging_res2:
                    tag = index_based_tagging_res2[i]
                    print("Changed")
                    break
            start_idx = start_idx + len(wd) + 1
            # res = (pmid, wd, tag)
            # res = ''' "{}", "{}", "{}"'''.format(pmid, wd, tag)
            # res = '''{},{},{}'''.format(pmid, wd, tag)
            res = [pm_id, wd, tag]
            # print(res)
            final_res.append(res)
            for wdr in wd.split(" "):
                wdr = clean_white_space(wdr)
                if wdr.strip() == "":
                    continue
                word_tag_obj.write("{} {}\n".format(wdr, tag))

    # new_csv = open("tagged_data.json", "w", encoding="utf8")
    # json.dump(final_res, new_csv)
    # new_csv.close()
    word_tag_obj.close()
    return


def main():
    gen_tags(test)
    gen_tags(train)
    gen_tags(dev)


if __name__ == '__main__':
    main()
