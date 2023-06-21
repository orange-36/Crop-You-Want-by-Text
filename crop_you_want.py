import sys
submodule_path = "GroundingDINO/"
sys.path.append(submodule_path)
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
from pathlib import Path
import os
import argparse
import torch
import cv2

from collections import defaultdict

import numpy as np
from torchvision.ops import box_convert



def get_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--image_path', '-i', type=Path, nargs= '+', required=True, help='The input image path, you can input multiple paths or the directory path containing images to be processed.')
    parser.add_argument('--text_prompt', '-t', type=str, required=True, help='Enter a text prompt, which can be a word or a phrase. Use ''." to separate different text categories.')
    parser.add_argument('--box_threshold', default=0.25, type=float, required=False, help='Threshold for bounding box.')
    parser.add_argument('--text_threshold', default=0.25, type=float, required=False, help='Threshold to judge whether it is the corresponding text category.')
    parser.add_argument('--extend', default=0, type=int, required=False, help='Extra dilated target box to crop.')
    parser.add_argument('--model_config', default=os.path.join(submodule_path, "groundingdino/config/GroundingDINO_SwinT_OGC.py"), type=Path, required=False, help='model_config')
    parser.add_argument('--model_weight', default="weights/groundingdino_swint_ogc.pth", type=Path, required=False, help='model_weight')
    parser.add_argument('--output_path', default="output/", type=Path, required=False, help='Where to save the results.')
    parser.add_argument('--device', default="cuda", type=str, required=False, help='Device want to use. If no gpu is available, set cpu.')
    parser.add_argument('--output_order', default="score", type=str, required=False, help='The order of the output results, set score to go from high to low according to the score, or set x or y to go from left to right or top to bottom.')
    parser.add_argument('--no_sub_dir', action="store_true", help='Do not create additional subdirectories, take effect by entering --no_sub_dir.')

    return parser.parse_args()


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = args.device
    print(f"use device: {device}")
    if len(args.image_path) > 1:
        image_path_list = args.image_path
    elif os.path.isdir(args.image_path[0]):
        image_path_list = [ os.path.join(args.image_path[0], file) for file in os.listdir(args.image_path[0])]
        args.output_path = os.path.join(args.output_path, args.image_path[0])
    else:
        image_path_list = [args.image_path]

    TEXT_PROMPT = args.text_prompt
    BOX_TRESHOLD = args.box_threshold
    TEXT_TRESHOLD = args.text_threshold
    model = load_model(args.model_config, args.model_weight)

    counter = defaultdict(int)
    for IMAGE_PATH in image_path_list:
        ##### Rredict one image #####
        filename = os.path.splitext(os.path.basename(IMAGE_PATH))[0] 
        file_extension = os.path.splitext(os.path.basename(IMAGE_PATH))[1] 
        
        if args.no_sub_dir:
            output_dir_for_a_image = Path(args.output_path)
        else:
            output_dir_for_a_image = Path(os.path.join(args.output_path, filename))
            counter = defaultdict(int)

        image_source, image = load_image(IMAGE_PATH)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD, 
            device=device
        )

        ##### Save Rrediction #####
        if not os.path.exists(output_dir_for_a_image):
            output_dir_for_a_image.mkdir(parents=True, exist_ok=True)

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        cv2.imwrite(os.path.join(output_dir_for_a_image, f'{filename}_annotation{file_extension}'), annotated_frame)

        h, w, _ = image_source.shape
        image_source = image_source[:, :, ::-1] ## RBG to GBR
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        if args.output_order == "score":
            sorted_idx = sorted(range(len(logits)), key=lambda i:logits[i], reverse=True)
        elif args.output_order == "x":
            sorted_idx = sorted(range(len(xyxy_boxes)), key=lambda i:xyxy_boxes[i][0])
        elif args.output_order == "y":
            sorted_idx = sorted(range(len(xyxy_boxes)), key=lambda i:xyxy_boxes[i][1])


        for i in sorted_idx:
            box, label, score = xyxy_boxes[i], phrases[i], int(np.around(logits[i], 2)*100)
            x1 = int(max(box[0]-args.extend, 0))
            y1 = int(max(box[1]-args.extend, 0))
            x2 = int(min(np.ceil(box[2])+args.extend+1, w))
            y2 = int(min(np.ceil(box[3])+args.extend+1, h))

            crop_img = image_source[y1:y2, x1:x2]

            counter[label] += 1
            crop_image_name = f"{label}_{counter[label]}_{score}{file_extension}"
            cv2.imwrite(os.path.join(output_dir_for_a_image, crop_image_name), crop_img)

        print(f"{IMAGE_PATH} accumulation:\n\t{dict(counter)}")

if __name__ == '__main__':
    args = get_args()
    print(args)
    main()