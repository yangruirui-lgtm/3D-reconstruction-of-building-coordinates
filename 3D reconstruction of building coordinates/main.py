import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re


# ===================== 数据解析模块 =====================
def parse_cameras(file_path):
    """解析cameras.txt，返回相机参数字典"""
    cameras = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            # 解析：CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            parts = re.split(r'\s+', line)
            cam_id = int(parts[0])
            cameras[cam_id] = {
                'model': parts[1],
                'width': int(parts[2]),
                'height': int(parts[3]),
                'params': [float(p) for p in parts[4:]]
            }
    return cameras


def parse_points3d(file_path):
    """解析points3D.txt，返回3D点云数据"""
    points = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # 解析：POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
            parts = re.split(r'\s+', line)
            point_id = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            error = float(parts[7])
            track = [int(t) for t in parts[8:]] if len(parts) > 8 else []
            points.append({
                'id': point_id,
                'xyz': (x, y, z),
                'rgb': (r / 255, g / 255, b / 255),  # 归一化到0-1
                'error': error,
                'track': track
            })
    return points


def parse_images(file_path):
    """解析images.txt，返回图像位姿数据"""
    images = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        # 每两行对应一个图像：第一行是位姿，第二行是点坐标（示例中为0 0 0）
        for i in range(0, len(lines), 2):
            pose_line = lines[i]
            # 解析：IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            pose_parts = re.split(r'\s+', pose_line)
            img_id = int(pose_parts[0])
            qw, qx, qy, qz = float(pose_parts[1]), float(pose_parts[2]), float(pose_parts[3]), float(pose_parts[4])
            tx, ty, tz = float(pose_parts[5]), float(pose_parts[6]), float(pose_parts[7])
            cam_id = int(pose_parts[8])
            name = pose_parts[9] if len(pose_parts) > 9 else f'camera{img_id}.jpg'
            images[img_id] = {
                'quaternion': (qw, qx, qy, qz),  # 四元数（旋转）
                'translation': (tx, ty, tz),  # 平移向量
                'camera_id': cam_id,
                'name': name
            }
    return images


# ===================== 3D可视化模块 =====================
def visualize_3d_scene(cameras, points, images):
    """可视化3D点云 + 相机位姿"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 绘制3D点云
    if points:
        x = [p['xyz'][0] for p in points]
        y = [p['xyz'][1] for p in points]
        z = [p['xyz'][2] for p in points]
        colors = [p['rgb'] for p in points]
        ax.scatter(x, y, z, c=colors, s=20, alpha=0.8, label='3D Points')

    # 2. 绘制相机位姿（用球体表示相机位置）
    if images:
        cam_x = [img['translation'][0] for img in images.values()]
        cam_y = [img['translation'][1] for img in images.values()]
        cam_z = [img['translation'][2] for img in images.values()]
        ax.scatter(cam_x, cam_y, cam_z, c='red', s=100, marker='^',
                   label='Camera Poses', edgecolors='black')
        # 标注相机名称
        for img in images.values():
            tx, ty, tz = img['translation']
            ax.text(tx, ty, tz, img['name'], fontsize=8)

    # 设置坐标轴标签
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Reconstruction of Building Coordinates')
    ax.legend()
    plt.show()


# ===================== 主函数 =====================
if __name__ == '__main__':
    # 替换为你的文件路径
    CAMERAS_PATH = '3D reconstruction of building coordinates/data/cameras.txt'
    POINTS3D_PATH = '3D reconstruction of building coordinates/data/points3D.txt'
    IMAGES_PATH = '3D reconstruction of building coordinates/data/images.txt'

    # 解析数据
    print("解析相机参数...")
    cameras = parse_cameras(CAMERAS_PATH)
    print(f"解析到 {len(cameras)} 个相机")

    print("解析3D点云...")
    points = parse_points3d(POINTS3D_PATH)
    print(f"解析到 {len(points)} 个3D点")

    print("解析图像位姿...")
    images = parse_images(IMAGES_PATH)
    print(f"解析到 {len(images)} 个图像位姿")

    # 可视化3D场景
    print("生成3D可视化...")
    visualize_3d_scene(cameras, points, images)

    # 可选：输出关键信息示例
    print("\n=== 示例数据 ===")
    print("相机1参数:", cameras[1])
    print("3D点1坐标:", points[0]['xyz'])
    print("图像1位姿（平移）:", images[1]['translation'])