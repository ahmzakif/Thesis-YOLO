import skripsi
import shutil
import argparse

model = skripsi.InferNCNN(model_path='model/best11n_ncnn_model')

parser = argparse.ArgumentParser(description='Infer YOLO model')

parser.add_argument('--mode', type=str, help='Toggle Run Image/Video/Batch')
parser.add_argument('--path', type=str, help='Input Path')
parser.add_argument('--src', type=int, help='Input Source')
parser.add_argument('--vis', help='Toggle Visualize', default=False, dest='vis', action='store_true')
parser.add_argument('--txt', help='Toggle Write Box to .txt', default=False, dest='txt', action='store_true')

args = parser.parse_args()
if args.mode == 'image':
    model.run_image(args.path, enable_vis=args.vis, enable_write=args.txt)

elif args.mode == 'video':
    model.run_video(args.path, enable_vis=args.vis)

elif args.mode == 'webcam':
    model.run_webcam(args.src, enable_vis=args.vis)

# elif args.mode == 'prototype':
#     model.run(args.src, enable_vis=args.vis)

elif args.mode == 'batch':
    shutil.rmtree('./outputs', ignore_errors=True)
    model.run_batches(args.path)
