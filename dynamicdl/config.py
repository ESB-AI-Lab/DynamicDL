config = {
    "MAX_PBAR_DEPTH": 4,
    "VALID_ENTRY_SETS": [
        {'IMAGE_SET_ID', 'IMAGE_SET_NAME'},
        {'XMIN', 'XMAX', 'YMIN', 'YMAX', 'XMID', 'YMID', 'X1', 'X2', 'Y1', 'Y2', 'WIDTH', 'HEIGHT',
         'BBOX_CLASS_ID', 'BBOX_CLASS_NAME'},
        {'POLYGON', 'SEG_CLASS_ID', 'SEG_CLASS_NAME'},
        {'X', 'Y'}
    ],
    "MODES": {
        "inference": {'ABSOLUTE_FILE', 'IMAGE_ID', 'IMAGE_DIM'},
        "diffusion": {'ABSOLUTE_FILE', 'IMAGE_ID', 'IMAGE_DIM'},
        "classification": {'ABSOLUTE_FILE', 'IMAGE_ID', 'CLASS_ID', 'IMAGE_DIM'},
        "detection": {'ABSOLUTE_FILE', 'IMAGE_ID', 'BBOX_CLASS_ID', 'BOX', 'IMAGE_DIM'},
        "segmentation_mask": {'ABSOLUTE_FILE', 'IMAGE_ID', 'ABSOLUTE_FILE_SEG', 'IMAGE_DIM'},
        "segmentation_poly": {'ABSOLUTE_FILE', 'IMAGE_ID', 'POLYGON', 'SEG_CLASS_ID', 'IMAGE_DIM'}
    },
    "BBOX_MODES": [
        (
            {'X1', 'X2', 'Y1', 'Y2'},
            ('X1', 'Y1', 'X2', 'Y2'),
            (lambda x: round(min(x[0], x[1]), 6), lambda y: round(min(y[0], y[1]), 6),
             lambda x: round(max(x[0], x[1]), 6), lambda y: round(max(y[0], y[1]), 6))
        ),
        (
            {'XMIN', 'YMIN', 'XMAX', 'YMAX'},
            ('XMIN', 'YMIN', 'XMAX', 'YMAX'),
            (lambda x: round(x[0], 6), lambda y: round(y[0], 6),
             lambda x: round(x[1], 6), lambda y: round(y[1], 6))
        ),
        (
            {'XMIN', 'YMIN', 'WIDTH', 'HEIGHT'},
            ('XMIN', 'YMIN', 'WIDTH', 'HEIGHT'),
            (lambda x: round(x[0], 6), lambda y: round(y[0], 6),
             lambda x: round(x[0]+x[1], 6), lambda y: round(y[0]+y[1], 6))
        ),
        (
            {'XMID', 'YMID', 'WIDTH', 'HEIGHT'},
            ('XMID', 'YMID', 'WIDTH', 'HEIGHT'),
            (lambda x: round(x[0]-x[1]/2, 6), lambda y: round(y[0]-y[1]/2, 6),
             lambda x: round(x[0]+x[1]/2, 6), lambda y: round(y[0]+y[1]/2, 6))
        ),
        (
            {'XMAX', 'YMAX', 'WIDTH', 'HEIGHT'},
            ('XMAX', 'YMAX', 'WIDTH', 'HEIGHT'),
            (lambda x: round(x[0]-x[1], 6), lambda y: round(y[0]-y[1], 6),
             lambda x: round(x[0], 6), lambda y: round(y[0], 6))
        )
    ],
    "BBOX_COLS": {'X1', 'X2', 'Y1', 'Y2', 'XMIN', 'XMAX', 'YMIN', 'YMAX', 'XMID', 'YMID', 'WIDTH',
                  'HEIGHT', 'BBOX_CLASS_ID', 'BBOX_CLASS_NAME'}
}
