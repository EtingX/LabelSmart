import os
import cv2
import random
import xml.etree.ElementTree as ET
from lxml import etree, objectify
import argparse
import numpy as np


class ObjectDetectionCrop():
    def __init__(self, is_crop_img_bboxes=True,length=(2048, 2048)):

        # 配置各个操作的属性
        self.is_crop_img_bboxes = is_crop_img_bboxes
        self.length = length

    def isIntersection(self, xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
        intersect_flag = True

        minx = max(xmin_a, xmin_b)
        miny = max(ymin_a, ymin_b)

        maxx = min(xmax_a, xmax_b)
        maxy = min(ymax_a, ymax_b)
        if minx > maxx or miny > maxy:
            intersect_flag = False
        return intersect_flag

    def mat_inter(self, box1, box2):
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
        ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
        sax = abs(x01 - x02)
        sbx = abs(x11 - x12)
        say = abs(y01 - y02)
        sby = abs(y11 - y12)
        if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
            return True
        else:
            return False

    def _crop_img (self, img, bbox,at_least_one = True):
        satisfied_crop = False

        w = img.shape[0]
        h = img.shape[1]
        length = self.length
        limited_w = int(w - length[0])
        limited_h = int(h - length[1])
        if at_least_one:
            satisified_at_least = False
            while satisified_at_least == False:
                satisfied_crop = False
                while satisfied_crop == False:
                    wrong_intersection = 0
                    intersection_bbox = []
                    # print(limited_w)
                    # print(limited_h)
                    new_w = random.randint(1, limited_w)
                    new_h = random.randint(1, limited_h)

                    # 随机截图
                    cropImg = img[(new_w):(new_w + length[0]), (new_h):(new_h + length[1])]

                    crop_x_0 = new_h
                    crop_y_0 = new_w
                    crop_x_1 = new_h + length[1]
                    crop_y_1 = new_w + length[0]

                    crop_img_list = [crop_x_0, crop_y_0, crop_x_1, crop_y_1]

                    for box in bbox:
                        xmin1, ymin1, xmax1, ymax1 = box
                        xmin2, ymin2, xmax2, ymax2 = crop_img_list

                        intersection_box = [xmin1, ymin1, xmax1, ymax1]

                        isIntersection = self.mat_inter(box, crop_img_list)

                        if isIntersection == True:
                            # print('Intersection !')
                            # 计算相交矩形
                            xmin = max(xmin1, xmin2)
                            ymin = max(ymin1, ymin2)
                            xmax = min(xmax1, xmax2)
                            ymax = min(ymax1, ymax2)

                            w = max(0, xmax - xmin)
                            h = max(0, ymax - ymin)
                            area = w * h  # C∩G的面积
                            s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
                            if area == s1:
                                intersection_bbox.append(intersection_box)
                            else:
                                wrong_intersection = wrong_intersection + 1
                        #         print('No satisfied crop ...')
                        #
                        # else:
                        #     print('No intersection !')

                    if wrong_intersection == 0:
                        # print("No wrong intersection")
                        satisfied_crop = True
                    # else:
                    #     # print('Random again !!!')

                    # 裁剪后的boundingbox坐标计算
                    crop_bboxes = list()
                    # print(bbox)
                    # print(intersection_bbox)
                    if len(bbox) == 0:
                        satisified_at_least = True
                        print('This is a empty label image')
                    else:
                        for bbox in intersection_bbox:
                            crop_bboxes.append(
                                [bbox[0] - crop_x_0, bbox[1] - crop_y_0, bbox[2] - crop_x_0, bbox[3] - crop_y_0])
                        if len(crop_bboxes) != 0:
                            satisified_at_least = True
                            print('The cropped image contained at least one label !')
        else:
            while satisfied_crop == False:
                wrong_intersection = 0
                intersection_bbox = []
                # print(limited_w)
                # print(limited_h)
                new_w = random.randint(1, limited_w)
                new_h = random.randint(1, limited_h)

                # 随机截图
                cropImg = img[(new_w):(new_w + length[0]), (new_h):(new_h + length[1])]

                crop_x_0 = new_h
                crop_y_0 = new_w
                crop_x_1 = new_h + length[1]
                crop_y_1 = new_w + length[0]

                crop_img_list = [crop_x_0, crop_y_0, crop_x_1, crop_y_1]

                for box in bbox:
                    xmin1, ymin1, xmax1, ymax1 = box
                    xmin2, ymin2, xmax2, ymax2 = crop_img_list

                    intersection_box = [xmin1, ymin1, xmax1, ymax1]

                    isIntersection = self.mat_inter(box, crop_img_list)

                    if isIntersection == True:
                        # print('Intersection !')
                        # 计算相交矩形
                        xmin = max(xmin1, xmin2)
                        ymin = max(ymin1, ymin2)
                        xmax = min(xmax1, xmax2)
                        ymax = min(ymax1, ymax2)

                        w = max(0, xmax - xmin)
                        h = max(0, ymax - ymin)
                        area = w * h  # C∩G的面积
                        s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
                        if area == s1:
                            intersection_bbox.append(intersection_box)
                        else:
                            wrong_intersection = wrong_intersection + 1
                    #         print('No satisfied crop ...')
                    #
                    # else:
                    #     print('No intersection !')

                if wrong_intersection == 0:
                    print("No wrong intersection")
                    satisfied_crop = True
                # else:
                #     # print('Random again !!!')

                # 裁剪后的boundingbox坐标计算
                crop_bboxes = list()
                # print(bbox)
                # print(intersection_bbox)
                for bbox in intersection_bbox:
                    crop_bboxes.append(
                        [bbox[0] - crop_x_0, bbox[1] - crop_y_0, bbox[2] - crop_x_0, bbox[3] - crop_y_0])
                print('Normal cropped image !')

        return cropImg, crop_bboxes

        # 图像增强方法
    def dataAugment(self, img, bboxes):
        '''
        图像增强
        输入:
            img:图像array
            bboxes:该图像的所有框坐标
        输出:
            img:增强后的图像
            bboxes:增强后图片对应的box
        '''
        change_num = 0  # 改变的次数
        # print('------')
        while change_num < 1:  # 默认至少有一种数据增强生效
            if self.is_crop_img_bboxes:
                img, bboxes = self._crop_img(img, bboxes)
                change_num += 1

        return img, bboxes


class ToolHelper():
    # 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    def parse_xml(self, path):
        '''
        输入：
            xml_path: xml的文件路径
        输出：
            从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
        '''
        tree = ET.parse(path)
        root = tree.getroot()
        objs = root.findall('object')
        coords = list()
        for ix, obj in enumerate(objs):
            name = obj.find('name').text
            box = obj.find('bndbox')
            x_min = int(box[0].text)
            y_min = int(box[1].text)
            x_max = int(box[2].text)
            y_max = int(box[3].text)
            coords.append([x_min, y_min, x_max, y_max, name])
        return coords

    # 保存图片结果
    def save_img(self, file_name, save_folder, img):
        cv2.imwrite(os.path.join(save_folder, file_name), img)

    # 保持xml结果
    def save_xml(self, file_name, save_folder, img_info, height, width, channel, bboxs_info):
        '''
        :param file_name:文件名
        :param save_folder:#保存的xml文件的结果
        :param height:图片的信息
        :param width:图片的宽度
        :param channel:通道
        :return:
        '''
        folder_name, img_name = img_info  # 得到图片的信息

        E = objectify.ElementMaker(annotate=False)

        anno_tree = E.annotation(
            E.folder(folder_name),
            E.filename(img_name),
            E.path(os.path.join(folder_name, img_name)),
            E.source(
                E.database('Unknown'),
            ),
            E.size(
                E.width(width),
                E.height(height),
                E.depth(channel)
            ),
            E.segmented(0),
        )

        labels, bboxs = bboxs_info  # 得到边框和标签信息
        for label, box in zip(labels, bboxs):
            anno_tree.append(
                E.object(
                    E.name(label),
                    E.pose('Unspecified'),
                    E.truncated('0'),
                    E.difficult('0'),
                    E.bndbox(
                        E.xmin(box[0]),
                        E.ymin(box[1]),
                        E.xmax(box[2]),
                        E.ymax(box[3])
                    )
                ))

        etree.ElementTree(anno_tree).write(os.path.join(save_folder, file_name), pretty_print=True)


# 显示图片
def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)
    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200, 800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    need_aug_num = 1  # 每张图片需要增强的次数

    is_endwidth_dot = True  # 文件是否以.jpg或者png结尾

    dataAug = ObjectDetectionCrop()  # 数据增强工具类

    toolhelper = ToolHelper()  # 工具

    # 获取相关参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_img_path', type=str, default='I:\canola_2022_project/fpis_test/resize_test/test/15\images/test')
    parser.add_argument('--source_xml_path', type=str, default='I:\canola_2022_project/fpis_test/resize_test/test/15/voc_label')
    parser.add_argument('--save_img_path', type=str, default='crop/images')
    parser.add_argument('--save_xml_path', type=str, default='crop/labels')

    args = parser.parse_args()
    source_img_path = args.source_img_path  # 图片原始位置
    source_xml_path = args.source_xml_path  # xml的原始位置

    save_img_path = args.save_img_path  # 图片增强结果保存文件
    save_xml_path = args.save_xml_path  # xml增强结果保存文件

    # 如果保存文件夹不存在就创建
    if not os.path.exists(save_img_path):
        os.mkdir(save_img_path)

    if not os.path.exists(save_xml_path):
        os.mkdir(save_xml_path)

    for parent, _, files in os.walk(source_img_path):
        files.sort()
        for file in files:
            cnt = 0
            pic_path = os.path.join(parent, file)
            xml_path = os.path.join(source_xml_path, file[:-4] + '.xml')
            values = toolhelper.parse_xml(xml_path)  # 解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]
            coords = [v[:4] for v in values]  # 得到框
            labels = [v[-1] for v in values]  # 对象的标签

            # 如果图片是有后缀的
            if is_endwidth_dot:
                # 找到文件的最后名字
                dot_index = file.rfind('.')
                _file_prefix = file[:dot_index]  # 文件名的前缀
                _file_suffix = file[dot_index:]  # 文件名的后缀
            img = cv2.imread(pic_path)

            # show_pic(img, coords)  # 显示原图
            while cnt < need_aug_num:  # 继续增强
                auged_img, auged_bboxes = dataAug.dataAugment(img, coords)
                auged_bboxes_int = np.array(auged_bboxes).astype(np.int32)
                height, width, channel = auged_img.shape  # 得到图片的属性
                img_name = '{}_{}{}'.format(_file_prefix, cnt + 1, _file_suffix)  # 图片保存的信息
                toolhelper.save_img(img_name, save_img_path,
                                    auged_img)  # 保存增强图片

                toolhelper.save_xml('{}_{}.xml'.format(_file_prefix, cnt + 1),
                                    save_xml_path, (save_img_path, img_name), height, width, channel,
                                    (labels, auged_bboxes_int))  # 保存xml文件
                # show_pic(auged_img, auged_bboxes)  # 强化后的图
                print(img_name)
                cnt += 1  # 继续增强下一张