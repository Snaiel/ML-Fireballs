import cv2
path_to_img = "../data/GFO_fireball_object_detection_training_set/jpegs/25_2016-08-22_202428_S_DSC_2787.thumb.jpg"
img = cv2.imread(path_to_img)
img_h, img_w, _ = img.shape
split_width = 400
split_height = 400


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

print(X_points, Y_points, len(X_points), len(Y_points))
print("Total tiles:", len(X_points) * len(Y_points))

count = 0
name = 'splitted'
frmt = 'jpeg'

for i in Y_points:
    print(count)
    for j in X_points:
        split = img[i:i+split_height, j:j+split_width]
        cv2.imwrite(f"{name}_{i / split_height :.1f}_{j / split_width :.1f}.{frmt}", split)
        count += 1