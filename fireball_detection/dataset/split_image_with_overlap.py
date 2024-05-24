import cv2
path_to_img = "../data/GFO_fireball_object_detection_training_set/jpegs/03_2016-07-28_043558_K_DSC_8287.thumb.jpg"
img = cv2.imread(path_to_img)
img_h, img_w, _ = img.shape
split_width = 1840
split_height = 1228


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            if split_size == size:
                break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


X_points = start_points(img_w, split_width, 0.5)
Y_points = start_points(img_h, split_height, 0.5)

print("Total tiles:", X_points * Y_points)

count = 0
name = 'splitted'
frmt = 'jpeg'

# for i in Y_points:
#     print(count)
#     for j in X_points:
#         split = img[i:i+split_height, j:j+split_width]
#         cv2.imwrite('{}_{}.{}'.format(name, count, frmt), split)
#         count += 1