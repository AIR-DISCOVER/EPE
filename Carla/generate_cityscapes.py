# This file should only be used once.
# It takes train_images.txt, train_labels.txt, val_images.txt, val_labels.txt as input
# And output an output.txt with the following format:
# image_dir,gtlabel_dir

if __name__ == '__main__':
    train_images_txt = "/home/share/datasets/cityscapes/train_images.txt"
    train_labels_txt = "/home/share/datasets/cityscapes/train_labels.txt"
    val_images_txt = "/home/share/datasets/cityscapes/val_images.txt"
    val_labels_txt = "/home/share/datasets/cityscapes/val_labels.txt"

    output_txt = "/home/gaoha/epe/Carla/cityscapes.txt"

    f = open(output_txt, 'w+', encoding='utf-8')
    g = open(train_images_txt, 'r', encoding='utf-8')
    h = open(train_labels_txt, 'r', encoding='utf-8')

    line_g = len(g.readlines())
    g = open(train_images_txt, 'r', encoding='utf-8')

    for i in range(line_g):
        f.write('/home/share/datasets/cityscapes/' + g.readline().strip() + ',' + '/home/share/datasets/cityscapes/' + h.readline().strip().replace('trainLabelIds.png', 'labelIds.png') + '\n')
    g.close()
    h.close()

    g = open(val_images_txt, 'r', encoding='utf-8')
    h = open(val_labels_txt, 'r', encoding='utf-8')

    line_g = len(g.readlines())
    g = open(val_images_txt, 'r', encoding='utf-8')
    for i in range(line_g):
        f.write(
            '/home/share/datasets/cityscapes/' + g.readline().strip() + ',' + '/home/share/datasets/cityscapes/' + h.readline().strip().replace('trainLabelIds.png', 'labelIds.png') + '\n')
    g.close()
    h.close()

    f.close()
