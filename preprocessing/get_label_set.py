def get_label_txt(unique_labels):
    label_set = open('labels.txt', 'a')

    for r1 in unique_labels:
        label_set.write((b'\033$B' + bytes.fromhex(str(hex(r1))[2:])).decode('iso2022_jp') + '\n')
    label_set.close()
