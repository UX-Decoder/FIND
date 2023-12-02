import json

# entity_val2017.json, entity_val2017_long.json, entity_train2017.json
annot_root = '/nobackup3/xueyan-data/grin_data/coco/annotations/entity_val2017_long.json'
annotations = json.load(open(annot_root, 'r'))

print("image number: {}".format(len(annotations['images'])))
print("caption number: {}".format(len(annotations['annotations'])))

entity_count = 0
for annot in annotations['annotations']:
    entity_count += len(annot['phrase'])

print("entity number: {}".format(entity_count))