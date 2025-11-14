import json
import numpy as np
import os

def read_cameras_txt(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            l = line.strip()
            if l.startswith("#") or len(l) == 0:
                continue
            elems = l.split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = [float(x) for x in elems[4:]]
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params,
            }
    return cameras

def read_images_txt(path):
    images = {}
    image_order = []
    with open(path) as f:
        lines = f.readlines()
    idx = 0
    while idx < len(lines):
        l = lines[idx].strip()
        if l.startswith("#") or len(l) == 0:
            idx += 1
            continue
        elems = l.split()
        image_id = int(elems[0])
        qw, qx, qy, qz = map(float, elems[1:5])
        tx, ty, tz = map(float, elems[5:8])
        camera_id = int(elems[8])
        name = elems[9]
        # The line after this has 2D-3D correspondences, which we ignore here
        images[image_id] = {
            'qvec': [qw, qx, qy, qz],
            'tvec': [tx, ty, tz],
            'camera_id': camera_id,
            'file_path': name,
        }
        image_order.append(image_id)
        idx += 2  # Skip 2 lines per image
    return images, image_order

def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    R = np.array([
        [1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw, 1-2*qx**2-2*qz**2, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx**2-2*qy**2]
    ])
    return R

def make_intrinsic(model, params):
    # Assuming simplest pinhole/radial model; may need to adapt for others
    # "heikkila" style (fx, fy, cx, cy, distortion...) 
    if len(params) == 4:
        fx, fy, cx, cy = params
        distortion = [0, 0, 0, 0, 0]
    else:
        fx, fy, cx, cy = params[:4]
        distortion = params[4:9] if len(params) > 4 else [0, 0, 0, 0, 0]
    return {
        "focal": [fx, fy],
        "ppt": [cx, cy],
        "distortion param": distortion,
    }

def make_extrinsic(qvec, tvec):
    R = qvec2rotmat(qvec)
    t = np.array(tvec).reshape(3, 1)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t[:, 0]
    return extrinsic.flatten().tolist()

def main(colmap_sparse_dir, output_json, image_dir=None):
    cameras = read_cameras_txt(os.path.join(colmap_sparse_dir, 'cameras.txt'))
    images, image_order = read_images_txt(os.path.join(colmap_sparse_dir, 'images.txt'))
    
    # Step 1: image paths - create dict with string keys
    image_file_list = {}
    for idx, image_id in enumerate(sorted(image_order)):
        image_name = images[image_id]['file_path']
        if image_dir is not None:
            img_path = os.path.join(image_dir, image_name)
        else:
            img_path = image_name
        image_file_list[str(idx)] = img_path
    
    # Step 2: bbox (set to identity if unknown)
    bbox = {
        "transform": [1 if i%5==0 else 0 for i in range(16)]
    }
    
    # Step 3: build camera_track_map
    camera_track_map = {"images": {}}
    for idx, image_id in enumerate(sorted(image_order)):
        img_info = images[image_id]
        cam_info = cameras[img_info['camera_id']]
        intrinsic = make_intrinsic(cam_info['model'], cam_info['params'])
        extrinsic = make_extrinsic(img_info['qvec'], img_info['tvec'])
        
        # Nest intrinsic and extrinsic under 'camera' key
        camera_entry = {
            "camera": {
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
            },
            "flg": 2,
            "size": [cam_info['width'], cam_info['height']],
            "cid": img_info['camera_id'],
        }
        camera_track_map["images"][str(idx)] = camera_entry
    
    json_obj = {
        "image_path": {"file_paths": image_file_list},
        "bbox": bbox,
        "camera_track_map": camera_track_map
    }
    
    with open(output_json, "w") as f:
        json.dump(json_obj, f, indent=2)
    print(f"Wrote {output_json}")

if __name__ == '__main__':
    # main("PATH_TO_SPARSE", "sfm_scene.json", image_dir="PATH_TO_IMAGES")
    main("datasets/nike_virtual_colmaped/sparse/0", "sfm_scene.json", image_dir="datasets/nike_virtual_colmaped/images")

