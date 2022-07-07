# YOLOv6s model
model = dict(
    type='YOLOv6s',
    pretrained='runs/train/exp18/weights/last_ckpt.pt',
    depth_multiple=0.33,
    width_multiple=0.50,
    backbone=dict(
        type='EfficientRep',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        ),
    neck=dict(
        type='RepPAN',
        num_repeats=[12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512],
        ),
    head=dict(
        type='EffiDeHead',
        in_channels=[128, 256, 512],
        num_layers=3,
        begin_indices=24,
        anchors=1,
        out_indices=[17, 20, 23],
        strides=[8, 16, 32],
        iou_type='siou'
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.008,
    lrf=0.008,
    momentum=0.99,
    weight_decay=0.0003,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.07
)

data_aug = dict(
    hsv_h=0.0138,
    hsv_s=0.664,
    hsv_v=0.464,
    degrees=0.373,
    translate=0.245,
    scale=0.898,
    shear=0.602,
    flipud=0.00856,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.243,
)
