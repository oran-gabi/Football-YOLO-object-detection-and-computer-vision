# bbox_utils.py
def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def get_bbox_height(bbox):
    """Calculate the height of a bounding box.
    
    Args:
        bbox (list or tuple): The bounding box coordinates [x1, y1, x2, y2].
    
    Returns:
        int: The height of the bounding box.
    """
    return bbox[3] - bbox[1]  # Assuming bbox is [x1, y1, x2, y2]