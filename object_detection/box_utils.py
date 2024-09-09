def box_area(xyxy):
    """Calculate the area of a bounding box given in xyxy format."""
    x_min, y_min, x_max, y_max = xyxy
    width = max(0, x_max - x_min)
    height = max(0, y_max - y_min)
    return width * height


def iou(box1, box2):
    """Calculate the intersection over union box area.
    
    Args:
        box1: A tuple of (x_min, y_min, x_max, y_max) for the first box.
        box2: A tuple of (x_min, y_min, x_max, y_max) for the second box.

    Returns:
        A float representing the intersection over the union area of the two boxes.
    """

    # Calculate the coordinates of the intersection
    inter_x_min = max(box1[0], box2[0])
    inter_y_min = max(box1[1], box2[1])
    inter_x_max = min(box1[2], box2[2])
    inter_y_max = min(box1[3], box2[3])

    # Check if the intersection box is valid
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0
    
    # Calculate intersection area
    intersection_area = box_area((inter_x_min, inter_y_min, inter_x_max, inter_y_max))
    
    # Calculate the area of both boxes
    area_box1 = box_area(box1)
    area_box2 = box_area(box2)
    
    # Calculate the area of the union
    union_area = area_box1 + area_box2 - intersection_area
    
    # To avoid division by zero, return 0 if union_area is zero
    if union_area < 1e-5:
        return 0.0

    # Calculate intersection over union
    result = intersection_area / union_area
    return result



def iom(box1, box2):
    """Calculate the intersection over minimum box area.
    
    Args:
        box1: A tuple of (x_min, y_min, x_max, y_max) for the first box.
        box2: A tuple of (x_min, y_min, x_max, y_max) for the second box.

    Returns:
        A float representing the intersection over the smaller box's area.
    """

    # Calculate the coordinates of the intersection
    inter_x_min = max(box1[0], box2[0])
    inter_y_min = max(box1[1], box2[1])
    inter_x_max = min(box1[2], box2[2])
    inter_y_max = min(box1[3], box2[3])

    # Check if the intersection box is valid
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return 0.0
    
    # Calculate intersection area
    intersection_area = box_area((inter_x_min, inter_y_min, inter_x_max, inter_y_max))
    
    # Calculate the area of both boxes
    area_box1 = box_area(box1)
    area_box2 = box_area(box2)
    
    # Identify the smaller area
    smaller_area = min(area_box1, area_box2)
    
    # To avoid division by zero, return 0 if smaller_area is zero
    if smaller_area < 1e-5:
        return 0.0

    # Calculate intersection over the area of the smaller box
    result = intersection_area / smaller_area
    return result