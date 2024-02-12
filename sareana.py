import cv2
import numpy as np
from getMGRS import getMGRS

min_lat = -90
max_lat = 90
min_lon = -180
max_lon = 180
lon_per_pix = (max_lon - min_lon) / 21600
lat_per_pix = (max_lat - min_lat) / 10800

def sareana(cloud_im_path, saliency_im_path, grid):
    cloud_im = cv2.imread(cloud_im_path)
    cloud_im = ~cv2.cvtColor(cloud_im, cv2.COLOR_BGR2GRAY)
    cv2.imshow('cloud_im', cloud_im)
    cv2.waitKey(0)
    saliency_im = cv2.imread(saliency_im_path)
    saliency_im = cv2.cvtColor(saliency_im, cv2.COLOR_BGR2GRAY)
    im_height, im_width = saliency_im.shape[:2]
    cloud_im_resized = cv2.resize(cloud_im, (im_width, im_height))
    sareana_im = np.zeros((im_height, im_width, 3), dtype=np.uint8)
    sareana_im[:, :, 0] = cloud_im_resized
    sareana_im[:, :, 1] = saliency_im
    cv2.imwrite('sareana.jpg', sareana_im)

    reg_sareana = sareana_im.copy().astype('float32')
    reg_sareana = cv2.cvtColor(reg_sareana, cv2.COLOR_BGR2GRAY)
    reg_sals = {}
    for key, value in grid.items():
        left, bottom, right, top = value
        left = left - min_lon
        right = right - min_lon
        bottom = 180 - (bottom - min_lat)
        top = 180 - (top - min_lat)
        left_px = int(left/lon_per_pix)
        right_px = int(right/lon_per_pix)
        top_px = int(bottom/lat_per_pix)
        bottom_px = int(top/lat_per_pix)
        region_im = sareana_im[bottom_px:top_px, left_px:right_px]
        region_sal = region_im.sum() / (region_im.shape[0] * region_im.shape[1])
        reg_sals[key] = region_sal
        reg_sareana[bottom_px:top_px, left_px:right_px] = region_sal
    reg_sareana = reg_sareana / reg_sareana.max() * 255
    reg_sareana = reg_sareana.astype('uint8')
    cv2.imwrite('reg_sareana.jpg', reg_sareana)
    sorted_reg_sals = sorted(reg_sals.items(), key=lambda x: x[1], reverse=True)
    return sorted_reg_sals


if __name__ == '__main__':
    grid = getMGRS()
    sorted_reg_sals = sareana('consolidated_cloudmap.jpg', 'world_saliency.jpg', grid)
    print(sorted_reg_sals)
    np.save('sorted_region_saliencys.npy', sorted_reg_sals)
    