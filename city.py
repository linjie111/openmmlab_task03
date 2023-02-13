_base_ = ['pspnet_r50-d8_512x512_20k_voc12aug.py']

model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    decode_head=dict(
        num_classes=2,
    ),
    auxiliary_head=dict(
        num_classes=2,
    )
)


data_root = '/openbayes/input/input0/'


data = dict(
    
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='PascalVOCDataset',
        data_root='/openbayes/input/input0/',
        img_dir='src',
        ann_dir='label',
        split='/openbayes/home/splits/train.txt',
        
        
    ),
    val=dict(
        type='PascalVOCDataset',
        data_root='/openbayes/input/input0/test/',
        img_dir='src',
        ann_dir='label',
        split='/openbayes/home/splits/val.txt',
        
        
    ),
    test=dict(
        type='PascalVOCDataset',
        data_root='/openbayes/input/input0/test/',
        img_dir='src',
        ann_dir='label',
        split='/openbayes/home/splits/val.txt',
      
    )
)

optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)
