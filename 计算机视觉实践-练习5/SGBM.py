import cv2
import numpy as np
from matplotlib import pyplot as plt


def opencv_SGBM(left_img, right_img, use_wls=True, sgbm="param1"):
    channels = 1 if left_img.ndim == 2 else 3
    blockSize = 6
    if sgbm == "param1":
        paramL = {
            "minDisparity": 8,
            "numDisparities": 4 * 16,
            "blockSize": blockSize,
            "P1": 8 * 3 * blockSize,
            "P2": 32 * 3 * blockSize,
            "disp12MaxDiff": 12,
            "uniquenessRatio": 10,
            "speckleWindowSize": 50,
            "speckleRange": 32,
            "preFilterCap": 63,
            "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        }
    elif sgbm == "param2":
        paramL = {
            "minDisparity": 0,
            "numDisparities": 5 * 16,
            "blockSize": blockSize * 3,
            "P1": 8 * 3 * blockSize,
            "P2": 32 * 3 * blockSize,
            "disp12MaxDiff": 50,
            "uniquenessRatio": 6,
            "speckleWindowSize": 150,
            "speckleRange": 32,
            "preFilterCap": 63,
            "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        }
    else:
        paramL = {
            "minDisparity": 0,
            "numDisparities": 128,
            "blockSize": blockSize,
            "P1": 8 * channels * blockSize**2,
            "P2": 32 * channels * blockSize**2,
            "disp12MaxDiff": 1,
            "preFilterCap": 63,
            "uniquenessRatio": 15,
            "speckleWindowSize": 100,
            "speckleRange": 1,
            "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        }

    matcherL = cv2.StereoSGBM_create(**paramL)
    # 计算视差图
    dispL = matcherL.compute(left_img, right_img)
    dispL = np.int16(dispL)
    # WLS滤波平滑优化图像
    if use_wls:
        # paramR = paramL
        # paramR['minDisparity'] = -paramL['numDisparities']
        # matcherR = cv2.StereoSGBM_create(**paramR)
        matcherR = cv2.ximgproc.createRightMatcher(matcherL)
        dispR = matcherR.compute(right_img, left_img)
        dispR = np.int16(dispR)
        lmbda = 80000
        sigma = 1.3
        filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcherL)
        filter.setLambda(lmbda)
        filter.setSigmaColor(sigma)
        dispL = filter.filter(dispL, left_img, None, dispR)
        dispL = np.int16(dispL)
    # 除以16得到真实视差（因为SGBM算法得到的视差是×16的）
    dispL[dispL < 0] = 0
    dispL = dispL.astype(np.float32) / 16.0
    cv2.imwrite("../test.pfm", dispL)
    plt.imshow(dispL, "gray")
    plt.show()


if __name__ == "__main__":
    left = cv2.imread("imgs/Art/view1.png", 0)
    right = cv2.imread("imgs/Art/view5.png", 0)
    opencv_SGBM(left, right)
