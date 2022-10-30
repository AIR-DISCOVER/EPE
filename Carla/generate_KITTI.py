# This file should only be used once.
# It takes train_images.txt, train_labels.txt, val_images.txt, val_labels.txt as input
# And output an output.txt with the following format:
# image_dir,gtlabel_dir

if __name__ == '__main__':

    root_dir = "/home/gaoha/epe/KITTI/color_label_2d"

    train_images_txt = f"{root_dir}/train_images.txt"
    train_labels_txt = f"{root_dir}/train_labels.txt"
    val_images_txt = f"{root_dir}/val_images.txt"
    val_labels_txt = f"{root_dir}/val_labels.txt"

    output_txt = "/home/gaoha/epe/KITTI/KITTI.txt"

    f = open(output_txt, 'w+', encoding='utf-8')
    g = open(train_images_txt, 'r', encoding='utf-8')
    h = open(train_labels_txt, 'r', encoding='utf-8')

    line_g = len(g.readlines())
    g = open(train_images_txt, 'r', encoding='utf-8')

    for i in range(line_g):
        f.write(root_dir + '/' + g.readline().strip() + ',' + root_dir + '/' + h.readline().strip() + '\n')
    g.close()
    h.close()

    g = open(val_images_txt, 'r', encoding='utf-8')
    h = open(val_labels_txt, 'r', encoding='utf-8')

    line_g = len(g.readlines())
    g = open(val_images_txt, 'r', encoding='utf-8')
    for i in range(line_g):
        f.write(root_dir + '/' + g.readline().strip() + ',' + root_dir + '/' + h.readline().strip() + '\n')
    g.close()
    h.close()

    f.close()
