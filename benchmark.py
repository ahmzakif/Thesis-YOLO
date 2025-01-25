import argparse
import os
import shutil

import skripsi

model = skripsi.InferNCNN(model_path='model\yolo11n_v4_bs32_ncnn_model')

parser = argparse.ArgumentParser(description='Object Detection')
parser.add_argument('--path', type=str, help='Input Path')

def prepare_input(path):
    # Clear directories
    shutil.rmtree('./mAP-skripsi/input/detection-results', ignore_errors=True)
    shutil.rmtree('./mAP-skripsi/input/ground-truth', ignore_errors=True)
    shutil.rmtree('./mAP-skripsi/input/images-optional', ignore_errors=True)

    # Create directories
    os.makedirs('./mAP-skripsi/input/detection-results', exist_ok=True)
    os.makedirs('./mAP-skripsi/input/ground-truth', exist_ok=True)
    os.makedirs('./mAP-skripsi/input/images-optional', exist_ok=True)
    
    # Copy XML and JPG files
    for file in os.listdir(path):
        if file.endswith('.xml'):
            shutil.copy(os.path.join(path, file), './mAP-skripsi/input/ground-truth/')
        elif file.endswith('.jpg'):
            shutil.copy(os.path.join(path, file), './mAP-skripsi/input/images-optional/')
    
    # Run the conversion script
    os.system('python ./mAP-skripsi/scripts/extra/convert_gt_xml.py')

def run_batches(path):
    # Clear the outputs directory
    shutil.rmtree('./outputs', ignore_errors=True)
    os.makedirs('./outputs', exist_ok=True)
    
    # Run the model
    model.run_batches(path)
    
    # Copy detection results
    for file in os.listdir('./outputs'):
        shutil.copy(os.path.join('./outputs', file), './mAP-skripsi/input/detection-results/')
    
    # Run the conversion script
    os.system('python convert.py')
    os.system('python ./mAP-skripsi/scripts/extra/intersect-gt-and-dr.py')

if __name__ == '__main__':
    os.system('git clone https://github.com/ahmzakif/mAP-skripsi.git')
    args = parser.parse_args()
    benchmark_path = args.path
    prepare_input(benchmark_path)
    run_batches(benchmark_path)
    os.system('python ./mAP-skripsi/main.py -na')

    