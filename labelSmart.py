import os
from shutil import move, copy
from sklearn.model_selection import train_test_split
import glob
import json
import cv2
from xml.dom.minidom import Document
import xml.etree.ElementTree as ET
import torch
import numpy as np

class Label():
    def __init__(self, label_dir, img_dir, out_dir=None):
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.out_dir = out_dir

    def no_label_search (self):
        label_list = []
        img_list = []

        for label in os.listdir(self.label_dir):
            label_list.append(label[:-4])
        for img in os.listdir(self.img_dir):
            img_list.append(img[:-4])

        no_label_img_list = set(label_list)^set(img_list)

        print('No labeled image ID are ')
        print(no_label_img_list)

        out_dir = self.out_dir
        if out_dir == None:
            out_dir = './out_dir/no_label_output'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            out_dir = os.path.join(out_dir,'no_label_output')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        for no_label_img in no_label_img_list:
            img = no_label_img+'.jpg'
            src_path = os.path.join(self.img_dir,img)
            dst_path = os.path.join(out_dir,img)

            move(src_path, dst_path)

    def no_img_search(self):
        label_list = []
        img_list = []

        for label in os.listdir(self.label_dir):
            label_list.append(label[:-4])
        for img in os.listdir(self.img_dir):
            img_list.append(img[:-4])

        no_image_label_list = set(img_list) ^ set(label_list)

        print('No imaging label ID are ')
        print(no_image_label_list)

        out_dir = self.out_dir
        if out_dir == None:
            out_dir = './out_dir/no_img_output'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            out_dir = os.path.join(out_dir, 'no_img_output')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        for no_img_label in no_image_label_list:
            label = no_img_label + '.txt'
            src_path = os.path.join(self.label_dir, label)
            dst_path = os.path.join(out_dir, label)

            move(src_path, dst_path)


    def train_val (self, ratio=0.2, random_state=None, test=False):
        label_list = []
        img_list = []

        for label in os.listdir(self.label_dir):
            label_list.append(label[:-4])
        for img in os.listdir(self.img_dir):
            img_list.append(img[:-4])

        LabelinImg = [False for c in label_list if c not in img_list]
        ImginLabel = [False for c in img_list if c not in label_list]

        if LabelinImg == False or ImginLabel == False:
            print('Image or label not match !')

        else:
            print('Split process start !')
            out_dir = self.out_dir
            if out_dir == None:
                out_dir = './out_dir/split_dataset'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
            else:
                out_dir = os.path.join(out_dir, 'split_dataset')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

            img_train_dir = os.path.join(out_dir,'images', 'train')
            img_val_dir = os.path.join(out_dir, 'images', 'val')
            label_train_dir = os.path.join(out_dir,'labels', 'train')
            label_val_dir = os.path.join(out_dir, 'labels', 'val')

            if not os.path.exists(img_train_dir):
                os.makedirs(img_train_dir)
            if not os.path.exists(img_val_dir):
                os.makedirs(img_val_dir)
            if not os.path.exists(label_train_dir):
                os.makedirs(label_train_dir)
            if not os.path.exists(label_val_dir):
                os.makedirs(label_val_dir)


            if random_state == None:
                train_list, val_list = train_test_split(label_list, test_size=ratio)

            else:
                train_list, val_list = train_test_split(label_list, test_size=ratio, random_state=random_state)

            print('Train list number: ' + str(len(train_list)) + ' Val list number: ' + str(len(val_list)))


            for item in train_list:
                img_name = item+'.jpg'

                src_path = os.path.join(self.img_dir, img_name)
                dst_path = os.path.join(img_train_dir, img_name)

                copy(src_path,dst_path)

                label_name = item + '.txt'

                label_src_path = os.path.join(self.label_dir, label_name)
                label_dst_path = os.path.join(label_train_dir, label_name)

                copy(label_src_path,label_dst_path)

            if test == False:
                for item_val in val_list:
                    img_name = item_val + '.jpg'

                    src_path = os.path.join(self.img_dir, img_name)
                    dst_path = os.path.join(img_val_dir, img_name)

                    copy(src_path, dst_path)

                    label_name = item_val + '.txt'

                    label_src_path = os.path.join(self.label_dir, label_name)
                    label_dst_path = os.path.join(label_val_dir, label_name)

                    copy(label_src_path, label_dst_path)
            else:
                if random_state == None:
                    val_list, test_list = train_test_split(val_list, test_size=ratio)

                else:
                    val_list, test_list = train_test_split(val_list, test_size=ratio, random_state=random_state)

                img_test_dir = os.path.join(out_dir, 'images', 'test')
                label_test_dir = os.path.join(out_dir, 'labels', 'test')

                if not os.path.exists(img_test_dir):
                    os.makedirs(img_test_dir)
                if not os.path.exists(img_test_dir):
                    os.makedirs(img_test_dir)

                for item_val in val_list:
                    img_name = item_val + '.jpg'

                    src_path = os.path.join(self.img_dir, img_name)
                    dst_path = os.path.join(img_val_dir, img_name)

                    copy(src_path, dst_path)

                    label_name = item_val + '.txt'

                    label_src_path = os.path.join(self.label_dir, label_name)
                    label_dst_path = os.path.join(label_val_dir, label_name)

                    copy(label_src_path, label_dst_path)

                for item_test in test_list:
                    img_name = item_test + '.jpg'

                    src_path = os.path.join(self.img_dir, img_name)
                    dst_path = os.path.join(img_test_dir, img_name)

                    copy(src_path, dst_path)

                    label_name = item_test + '.txt'

                    label_src_path = os.path.join(self.label_dir, label_name)
                    label_dst_path = os.path.join(label_test_dir, label_name)

                    copy(label_src_path, label_dst_path)

            print('Split finish !!!')


    def label_replace(self, replaced_label, replaced_label_dir=None):
        '''

        :param replaced_label:
        {'1':0,
        '2':0}
        :param replaced_label_dir: '/output'
        :return:
        '''
        if replaced_label_dir == None:
            replaced_label_dir = self.label_dir

        if len(os.listdir(replaced_label_dir)) != len(glob.glob(os.path.join(replaced_label_dir, "*.txt"))):
            print('Some files are not label file, this function only support YOLO label format !'
                  'You can use label convert to change your label format !')
        else:
            print('Replace label process start !')

            out_dir = self.out_dir
            if out_dir == None:
                out_dir = './out_dir/replaced_label'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
            else:
                out_dir = os.path.join(out_dir,'replaced_label')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

            key_list = list(replaced_label.keys())

            for label in os.listdir(replaced_label_dir):
                new_label = []
                label_path = os.path.join(replaced_label_dir, label)

                for line in open(label_path):
                    line_list = line.split(' ')
                    if line_list[0] in key_list:
                        line_list[0] = str(replaced_label[line_list[0]])
                    str_1 = ' '
                    new_line = str_1.join(line_list)
                    new_label.append(new_line)

                label_name = os.path.join(out_dir, label)
                yolo_txt = open(label_name, 'a')
                for label in new_label:
                    yolo_txt.write(label)

            print('Replace label process finish !!!')

    def label_remove(self, remove_label, remove_label_dir=None):
        num_remove = 0
        if remove_label_dir == None:
            remove_label_dir = self.label_dir

        if len(os.listdir(remove_label_dir)) != len(glob.glob(os.path.join(remove_label_dir, "*.txt"))):
            print('Some files are not label file, this function only support YOLO label format !'
                  'You can use label convert to change your label format !')
        else:
            print('Remove label process start !')
            out_dir = self.out_dir
            if out_dir == None:
                out_dir = './out_dir/removed_label'
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
            else:
                out_dir = os.path.join(out_dir,'removed_label')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

            for label in os.listdir(remove_label_dir):
                new_label = []
                label_path = os.path.join(remove_label_dir, label)

                for line in open(label_path):
                    line_list = line.split(' ')
                    if line_list[0] not in remove_label:
                        str_1 = ' '
                        new_line = str_1.join(line_list)
                        new_label.append(new_line)
                    elif line_list[0] in remove_label:
                        num_remove = num_remove + 1

                label_name = os.path.join(out_dir, label)
                yolo_txt = open(label_name, 'a')
                for label in new_label:
                    yolo_txt.write(label)

            print('Remove label process finish !!! Remove number: ' + str(num_remove))

    def label_select(self, select_label, select_label_dir=None):
        num_selected = 0
        if select_label_dir == None:
            select_label_dir = self.label_dir

        if len(os.listdir(select_label_dir)) != len(glob.glob(os.path.join(select_label_dir, "*.txt"))):
            print('Some files are not label file, this function only support YOLO label format !'
                  'You can use label convert to change your label format !')
        else:
            print('Selected label process start !')
            out_dir = self.out_dir
            selected_label_folder = 'selected_label_' + str(select_label)
            if out_dir == None:
                out_dir = os.path.join('./out_dir',selected_label_folder)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
            else:
                out_dir = os.path.join(out_dir,selected_label_folder)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

            for label in os.listdir(select_label_dir):
                label_path = os.path.join(select_label_dir, label)

                for line in open(label_path):
                    line_list = line.split(' ')
                    if line_list[0] == select_label:
                        src_path = os.path.join(self.label_dir, label)
                        dst_path = os.path.join(out_dir, label)

                        if not os.path.exists(dst_path):
                            copy(src_path, dst_path)

                            num_selected = num_selected + 1

            print('Selected label process finish !!! Select number: ' + str(num_selected))

    def emtpy_label(self, empty_img_dir = None):

        out_dir = self.out_dir
        if out_dir == None:
            out_dir = './out_dir/empty_label_output'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            out_dir = os.path.join(out_dir, 'empty_label_output')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        target_txt_path = out_dir
        if empty_img_dir == None:
            img_dir = self.img_dir
        else:
            img_dir = empty_img_dir

        img_list = os.listdir(img_dir)

        label_num = 0
        for img_name in img_list:
            txt = str(img_name[:-4])
            with open(os.path.join(target_txt_path, '{}.txt'.format(txt)), 'w', encoding='utf-8') as f:
                f.write('')
            label_num = label_num + 1

        print('Total image number is ' + str(len(img_list)) + ' and empty label is ' + str(label_num))

    def xywhn2xyxy(self, x, w, h, padw=0, padh=0):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
        y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
        y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
        y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
        return y

    def crop_img_yolo(self, img_path, label_path, save_path, img_id):
        if os.path.getsize(label_path) != 0:
            with open(label_path, 'r') as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                img = cv2.imread(img_path)

                h, w = img.shape[:2]

                lb[:, 1:] = self.xywhn2xyxy(lb[:, 1:], w, h, 0, 0)

                n = 0
                for _, x in enumerate(lb):
                    class_id = int(x[0])
                    if not os.path.exists(os.path.join(save_path,str(class_id))):
                        os.makedirs(os.path.join(save_path,str(class_id)))
                    x0 = int(x[1])
                    y0 = int(x[2])
                    x1 = int(x[3])
                    y1 = int(x[4])
                    crop = img[y0: y1, x0: x1]
                    if crop.size != 0:
                        save_img = str(img_id) + str(n) + '.JPG'
                        save = os.path.join(save_path,str(class_id), save_img)
                        cv2.imwrite(filename=save, img=crop)
                        n += 1

    def crop_img(self):
        print('Label format should be YOLO format')
        img_dir = self.img_dir
        label_dir = self.label_dir

        img_list = os.listdir(img_dir)

        out_dir = self.out_dir
        if out_dir == None:
            out_dir = './out_dir/crop_images'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        else:
            out_dir = os.path.join(out_dir, 'crop_images')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        for img in img_list:
            img_path = os.path.join(img_dir,img)
            label_name = img[:-4] + '.txt'

            label_path = os.path.join(label_dir,label_name)
            self.crop_img_yolo(img_path=img_path,label_path=label_path,save_path=out_dir, img_id=img[:-4])

            print(str(img) + ' done !')



class label_change():
    '''
    # voc2yolo: class_file = ['bud']
    # yolo2voc: class_file = {'0':'bud'}
    '''
    def __init__(self, label_dir, img_dir, save_dir=None):
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.save_dir = save_dir
        if self.save_dir == None:
            out_dir = 'label_change'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            self.save_dir = out_dir

    def creatML2Yolo(self, json_file, img, class_file, save_path):
        w, h, d = img.shape

        with open(json_file, 'r') as f:
            # loading creatML json
            json_label = json.load(f)
            img_name = json_label[0]['image'][:-4]

            # writing txt
            label_name = os.path.join(save_path, str(img_name + '.txt'))
            yolo_txt = open(label_name, 'a')

            for num_label in range(len(json_label[0]['annotations'])):
                label = json_label[0]['annotations'][num_label]["label"]
                label_id = class_file[label]
                x_yolo = round((json_label[0]['annotations'][num_label]["coordinates"]["x"]) / h, 6)
                y_yolo = round((json_label[0]['annotations'][num_label]["coordinates"]["y"]) / w, 6)
                w_yolo = round((json_label[0]['annotations'][num_label]["coordinates"]["width"]) / h, 6)
                h_yolo = round((json_label[0]['annotations'][num_label]["coordinates"]["height"]) / w, 6)
                writing = str(label_id) + ' ' + str(x_yolo) + ' ' + str(y_yolo) + ' ' + str(w_yolo) + ' ' + str(h_yolo)

                yolo_txt.write(writing)
                yolo_txt.write('\n')

            yolo_txt.close()


    def final_creatML2Yolo(self,class_file):
        img_path = self.img_dir
        label_path = self.label_dir
        save_path = self.save_dir
        class_file = class_file

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        files = glob.glob(label_path + "*.json")
        files = [i.split("\\")[-1].split(".json")[0] for i in files]

        for file in files:
            print(file)
            img = str(file) + '.JPG'
            json_file = str(file) + '.json'

            img = os.path.join(img_path, img)
            img = cv2.imread(img)
            json_file = os.path.join(label_path, json_file)

            self.creatML2Yolo(json_file, img, class_file, save_path)
            print('done !!!')


    def yolo2voc(self,class_file):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
        """此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
        """
        dic = class_file
        picPath = self.img_dir
        txtPath = self.label_dir
        xmlPath = os.path.join(self.save_dir, 'yolo2voc')
        if not os.path.exists(xmlPath):
            os.makedirs(xmlPath)

        files = os.listdir(txtPath)
        for i, name in enumerate(files):
            print(str(name))
            xmlBuilder = Document()
            annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
            xmlBuilder.appendChild(annotation)
            txtFile = open(os.path.join(txtPath,name))
            txtList = txtFile.readlines()
            img = cv2.imread(picPath + '/' + name[0:-4] + ".jpg")
            Pheight, Pwidth, Pdepth = img.shape

            folder = xmlBuilder.createElement("folder")  # folder标签
            foldercontent = xmlBuilder.createTextNode("driving_annotation_dataset")
            folder.appendChild(foldercontent)
            annotation.appendChild(folder)  # folder标签结束

            filename = xmlBuilder.createElement("filename")  # filename标签
            filenamecontent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
            filename.appendChild(filenamecontent)
            annotation.appendChild(filename)  # filename标签结束

            size = xmlBuilder.createElement("size")  # size标签
            width = xmlBuilder.createElement("width")  # size子标签width
            widthcontent = xmlBuilder.createTextNode(str(Pwidth))
            width.appendChild(widthcontent)
            size.appendChild(width)  # size子标签width结束

            height = xmlBuilder.createElement("height")  # size子标签height
            heightcontent = xmlBuilder.createTextNode(str(Pheight))
            height.appendChild(heightcontent)
            size.appendChild(height)  # size子标签height结束

            depth = xmlBuilder.createElement("depth")  # size子标签depth
            depthcontent = xmlBuilder.createTextNode(str(Pdepth))
            depth.appendChild(depthcontent)
            size.appendChild(depth)  # size子标签depth结束

            annotation.appendChild(size)  # size标签结束

            for j in txtList:
                oneline = j.strip().split(" ")
                object = xmlBuilder.createElement("object")  # object 标签
                picname = xmlBuilder.createElement("name")  # name标签
                namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
                picname.appendChild(namecontent)
                object.appendChild(picname)  # name标签结束

                pose = xmlBuilder.createElement("pose")  # pose标签
                posecontent = xmlBuilder.createTextNode("Unspecified")
                pose.appendChild(posecontent)
                object.appendChild(pose)  # pose标签结束

                truncated = xmlBuilder.createElement("truncated")  # truncated标签
                truncatedContent = xmlBuilder.createTextNode("0")
                truncated.appendChild(truncatedContent)
                object.appendChild(truncated)  # truncated标签结束

                difficult = xmlBuilder.createElement("difficult")  # difficult标签
                difficultcontent = xmlBuilder.createTextNode("0")
                difficult.appendChild(difficultcontent)
                object.appendChild(difficult)  # difficult标签结束

                bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
                xmin = xmlBuilder.createElement("xmin")  # xmin标签
                mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
                xminContent = xmlBuilder.createTextNode(str(mathData))
                xmin.appendChild(xminContent)
                bndbox.appendChild(xmin)  # xmin标签结束

                ymin = xmlBuilder.createElement("ymin")  # ymin标签
                mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
                yminContent = xmlBuilder.createTextNode(str(mathData))
                ymin.appendChild(yminContent)
                bndbox.appendChild(ymin)  # ymin标签结束

                xmax = xmlBuilder.createElement("xmax")  # xmax标签
                mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
                xmaxContent = xmlBuilder.createTextNode(str(mathData))
                xmax.appendChild(xmaxContent)
                bndbox.appendChild(xmax)  # xmax标签结束

                ymax = xmlBuilder.createElement("ymax")  # ymax标签
                mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
                ymaxContent = xmlBuilder.createTextNode(str(mathData))
                ymax.appendChild(ymaxContent)
                bndbox.appendChild(ymax)  # ymax标签结束

                object.appendChild(bndbox)  # bndbox标签结束

                annotation.appendChild(object)  # object标签结束

            f = open(xmlPath + '/' + name[0:-4] + ".xml", 'w')
            xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
            f.close()
            print('done !')


    def clear_hidden_files(self, path):
        dir_list = os.listdir(path)
        for i in dir_list:
            abspath = os.path.join(os.path.abspath(path), i)
            if os.path.isfile(abspath):
                if i.startswith("._"):
                    os.remove(abspath)
            else:
                self.clear_hidden_files(abspath)

    #数据转换
    def convert(self, size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)

    #编写格式
    def convert_annotation(self, xml_label, yolo_label, classes):
        '''

        :param xml_label:
        :param yolo_label:
        :param classes: ["class 1", 'class 2'] for example classes = ["boot", 'heading']
        :return:
        '''
        in_file = open(xml_label)
        out_file = open(yolo_label, 'w')
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = self.convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        in_file.close()
        out_file.close()


    def voc2yolo(self,class_file):
        '''
        :param classes: ["class 1", 'class 2'] for example classes = ["boot", 'heading']
        :return:
        '''
        label_dir = self.label_dir
        save_dir = os.path.join(self.save_dir,'voc2yolo')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        classes = class_file
        label_list = os.listdir(label_dir)
        for label in label_list:
            label_path = os.path.join(label_dir, label)
            label_name = str(label[:-4]) + '.txt'
            save_label = os.path.join(save_dir, label_name)
            self.convert_annotation(label_path, save_label, classes)