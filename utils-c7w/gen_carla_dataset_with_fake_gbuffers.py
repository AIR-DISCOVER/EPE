if __name__ == "__main__":

    original_file_txt = "/home/gaoha/epe/CarlaSampleDataset/file.txt"
    new_file_txt = "/home/gaoha/epe/CarlaSampleDataset/file_fake_gbuffers.txt"

    with open(original_file_txt, 'r') as f:
        lines = f.readlines()
        with open(new_file_txt, 'w') as f_new:
            for line in lines:
                line = line.strip()
                line_split = line.split(",")
                if not "sp2" in line_split[0]:
                    continue


                line_split[2] = "/home/gaoha/epe/code/Carla/OverfitTest/fake_gbuffers/" +\
                                line_split[0].split('/')[-1][:-len(".png")] + '.npz'
                print(line_split[2])

                line = ",".join(line_split)
                f_new.write(line + "\n")