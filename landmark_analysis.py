from multiprocessing import Pool, cpu_count
import cv2
import numpy as np

scales = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
consolidated_maps_path = 'bm1k_consolidated_maps'
prioritized_regions_path = 'prioritized_regions.csv'

def landmarks_at_scale(region, scale):
    im = cv2.imread('bm1k_consolidated_maps/' + region + '.jpg')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # kernel = np.ones((scale, scale), np.uint8) / (scale * scale)
    # old_sal = None
    overall_max = None
    landmark_list = []
    height, width = im.shape
    while(1):
        maxsum = 0
        for i in range(width-scale+1):
            for j in range(height-scale+1):
                cursum = im[j:j+scale, i:i+scale].sum()
                if cursum > maxsum:
                    maxsum = cursum
                    max_loc = (i, j) # x, y
        # filtered = cv2.filter2D(im, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        # filtered = filtered[scale:-scale, scale:-scale]
        # max_loc = filtered.argmax()
        # max_loc = np.unravel_index(max_loc, filtered.shape)
        # print(max_loc)
        # max_val = filtered.max()
        im[max_loc[1]:max_loc[1]+scale, max_loc[0]:max_loc[0]+scale] = 0
        # im[max_loc[0]:max_loc[0]+scale, max_loc[1]:max_loc[1]+scale] = 0
        # if old_sal is None:
        #     old_sal = maxsum
            # landmark_list.append([max_loc[0], max_loc[1], scale])
        # else:
        if overall_max is None:
            overall_max = maxsum
        if maxsum < 0.1*overall_max:
            break                
        # old_sal = maxsum
        landmark_list.append([max_loc[0], max_loc[1], scale])
    return landmark_list
        

if __name__ == '__main__': 
    # with open(prioritized_regions_path, 'r') as f:
    #     regions = f.read().split(',')
    # regions = [region for region in regions if region != '']
    # landmarks = landmarks_at_scale('17R', 50)
    # print(landmarks)
    region = '17R'
    pool = Pool(cpu_count())
    landmarks = pool.starmap(landmarks_at_scale, [(region, scale) for scale in scales])
    landmarks_array = np.concatenate(landmarks, axis=0)
    np.save(region+'_pixel_landmarks.npy', landmarks_array)
    pool.close()
    pool.join()