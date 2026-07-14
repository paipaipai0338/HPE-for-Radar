import torch
from roi_align import RoIAlign
from roi_align import CropAndResize

def RoIAlign_Fun(feature_map: torch.Tensor, boxes: torch.Tensor, crop_size: list):
    """
    feature_map: [B,C,H,W] or [B,T,C,H,W]
    boxes:       [B,4] or [B,T,4], format: [x1,y1,x2,y2], where
                 x indexes feature-map columns/W and y indexes rows/H.
    """
    restore_temporal = feature_map.dim() == 5

    if restore_temporal:
        B, T, C, H, W = feature_map.shape

        if boxes.shape != (B, T, 4):
            raise ValueError(
                f"Expected boxes [{B},{T},4], got {tuple(boxes.shape)}"
            )

        feature_map = feature_map.reshape(B * T, C, H, W)
        boxes = boxes.reshape(B * T, 4)

    elif feature_map.dim() == 4:
        B, C, H, W = feature_map.shape

        if boxes.shape != (B, 4):
            raise ValueError(
                f"Expected boxes [{B},4], got {tuple(boxes.shape)}"
            )
    else:
        raise ValueError(
            f"Expected 4D or 5D feature_map, got {feature_map.dim()}D"
        )

    if feature_map.dtype != torch.float32:
        raise TypeError("Current ROIAlign CUDA extension only supports float32")

    boxes = boxes.to(
        device=feature_map.device,
        dtype=torch.float32,
    ).contiguous()
    feature_map = feature_map.contiguous()

    num_boxes = boxes.shape[0]
    box_index = torch.arange(
        num_boxes,
        device=feature_map.device,
        dtype=torch.int32,
    )

    crop_height, crop_width = crop_size
    crops = RoIAlign(crop_height, crop_width)(
        feature_map,
        boxes,
        box_index,
    )

    if restore_temporal:
        crops = crops.reshape(
            B, T, C, crop_height, crop_width
        )

    return crops

if __name__ == '__main__':
    # input feature maps (suppose that we have batch_size==2)
    image = torch.arange(0., 49).view(1, 1, 7, 7).repeat(2, 1, 1, 1)
    image[0] += 50
    print('image: ', image.shape, image)


    # for example, we have two bboxes with coords xyxy (first with batch_id=0, second with batch_id=1).
    boxes = torch.Tensor([[1, 0, 5, 4],
                        [0.5, 3.5, 4, 7]])

    # box_index = torch.tensor([0, 1], dtype=torch.int) # index of bbox in batch

    
    crops = RoIAlign_Fun(image, boxes, (8,8))
    print('crops:', crops.shape, crops)
