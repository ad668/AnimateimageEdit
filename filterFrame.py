from tkinter import Toplevel, Button, RIGHT
import numpy as np
import cv2


class FilterFrame(Toplevel):

    def __init__(self, master=None):
        Toplevel.__init__(self, master=master)

        self.original_image = self.master.processed_image
        self.filtered_image = None

        button_style = {'background': 'blue', 'foreground': 'white', 'relief': 'ridge'}

        self.negative_button = Button(master=self, text="Negative", **button_style)
        self.black_white_button = Button(master=self, text="Black White", **button_style)
        self.sepia_button = Button(master=self, text="Sepia", **button_style)
        self.emboss_button = Button(master=self, text="Emboss", **button_style)
        self.gaussian_blur_button = Button(master=self, text="Gaussian Blur", **button_style)
        self.median_blur_button = Button(master=self, text="Median Blur", **button_style)
        self.details_button = Button(master=self, text="Details", **button_style)
        self.summer_button = Button(master=self, text="Summer", **button_style)
        self.winter_button = Button(master=self, text="Winter", **button_style)
        self.daylight_button = Button(master=self, text="DayLight", **button_style)
        self.grainy_button = Button(master=self, text="Grainy", **button_style)
        self.highcontrast_button=Button(master=self,text="High Contrast", **button_style)
        self.smoothen_button = Button(master=self, text="Smoothen", **button_style)
        self.vignette_button = Button(master=self, text="Vignette", **button_style)
        self.bonus_button = Button(master=self, text="Bonus", **button_style)
        self.distorted_button = Button(master=self, text="Distorted", **button_style)
        self.distorted_button = Button(master=self, text="Distorted", **button_style)
        self.cancel_button = Button(master=self, text="Cancel", **button_style)
        self.apply_button = Button(master=self, text="Apply", **button_style)
        self.details_button=Button(master=self,text="Details", **button_style)

        self.negative_button.bind("<ButtonRelease>", self.negative_button_released)
        self.black_white_button.bind("<ButtonRelease>", self.black_white_released)
        self.sepia_button.bind("<ButtonRelease>", self.sepia_button_released)
        self.emboss_button.bind("<ButtonRelease>", self.emboss_button_released)
        self.gaussian_blur_button.bind("<ButtonRelease>", self.gaussian_blur_button_released)
        self.median_blur_button.bind("<ButtonRelease>", self.median_blur_button_released)
        self.details_button.bind("<ButtonRelease>", self.details_button_released)
        self.summer_button.bind("<ButtonRelease>", self.summer_button_released)
        self.winter_button.bind("<ButtonRelease>", self.winter_button_released)
        self.daylight_button.bind("<ButtonRelease>", self.daylight_button_released)
        self.grainy_button.bind("<ButtonRelease>", self.grainy_button_released)
        self.smoothen_button.bind("<ButtonRelease>", self.smoothen_button_released)
        self.highcontrast_button.bind("<ButtonRelease>", self.highcontrast_button_released)
        self.distorted_button.bind("<ButtonRelease>", self.distorted_button_released)
        self.vignette_button.bind("<ButtonRelease>", self.vignette_button_released)
        self.bonus_button.bind("<ButtonRelease>", self.bonus_button_released)
        self.apply_button.bind("<ButtonRelease>", self.apply_button_released)
        self.cancel_button.bind("<ButtonRelease>", self.cancel_button_released)

        self.negative_button.pack()
        self.black_white_button.pack()
        self.sepia_button.pack()
        self.emboss_button.pack()
        self.gaussian_blur_button.pack()
        self.median_blur_button.pack()
        self.details_button.pack()
        self.summer_button.pack()
        self.winter_button.pack()
        self.distorted_button.pack()
        self.daylight_button.pack()
        self.grainy_button.pack()
        self.smoothen_button.pack()
        self.highcontrast_button.pack()
        self.vignette_button.pack()
        self.bonus_button.pack()
        self.cancel_button.pack(side=RIGHT)
        self.apply_button.pack()

    def details_button_released(self, event):
        self.details()
        self.show_image()
    def bonus_button_released(self, event):
        self.bonus()
        self.show_image()
    def distorted_button_released(self, event):
        self.distorted()
        self.show_image()
    def highcontrast_button_released(self, event):
        self.highcontrast()
        self.show_image()
    def summer_button_released(self, event):
        self.summer()
        self.show_image()
    def winter_button_released(self, event):
        self.winter()
        self.show_image()
    def daylight_button_released(self, event):
        self.daylight()
        self.show_image()
    def grainy_button_released(self, event):
        self.grainy()
        self.show_image()
    def smoothen_button_released(self, event):
        self.smoothen()
        self.show_image()
    def vignette_button_released(self, event):
        self.vignette()
        self.show_image()


    def details(self):

        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
        dst2 = cv2.filter2D(self.original_image, -1, kernel_sharpening)
        self.filtered_image=dst2

    def gamma_function(self,channel, gamma):
        invGamma = 1 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")  # creating lookup table
        channel = cv2.LUT(channel, table)
        return channel

    def summer(self):
        img = self.original_image
        img[:, :, 0] = self.gamma_function(img[:, :, 0], 0.75)  # down scaling blue channel
        img[:, :, 2] = self.gamma_function(img[:, :, 2], 1.25)  # up scaling red channel
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = self.gamma_function(hsv[:, :, 1], 1.2)  # up scaling saturation channel
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.filtered_image=img

    def winter(self):
        img = self.original_image
        img[:, :, 0] = self.gamma_function(img[:, :, 0], 1.25)  # down scaling blue channel
        img[:, :, 2] = self.gamma_function(img[:, :, 2], 0.75)  # up scaling red channel
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = self.gamma_function(hsv[:, :, 1], 0.8)  # up scaling saturation channel
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.filtered_image=img

    def vignette(self):
        rows, cols = self.original_image.shape[:2]

        X_resultant_kernel = cv2.getGaussianKernel(cols, 200)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, 200)

        resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T


        mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
        output = np.copy(self.original_image)


        for i in range(3):
            output[:, :, i] = output[:, :, i] * mask
        self.filtered_image=output

    def smoothen(self):
        dst2 = cv2.edgePreservingFilter(self.original_image, flags=1, sigma_s=60, sigma_r=0.4)
        self.filtered_image = dst2

    def bonus(self):
        dst = cv2.stylization(self.original_image, sigma_s=60, sigma_r=0.07)
        self.filtered_image = dst

    def distorted(self):
        dst_gray, dst_color = cv2.pencilSketch(self.original_image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        self.filtered_image = dst_color

    def daylight(self):
        img = self.original_image
        image_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        image_HLS = np.array(image_HLS, dtype=np.float64)
        daylight = 1.15
        image_HLS[:, :, 1] = image_HLS[:, :, 1] * daylight
        image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255
        image_HLS = np.array(image_HLS, dtype=np.uint8)
        image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2BGR)
        self.filtered_image=image_RGB

    def grainy(self):
        img = self.original_image
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = 0.8
        for i in range(height):
            for j in range(width):
                if np.random.rand() <= thresh:
                    if np.random.randint(2) == 0:
                        gray[i, j] = min(gray[i, j] + np.random.randint(0, 64),
                                         255)
                    else:
                        gray[i, j] = max(gray[i, j] - np.random.randint(0, 64),
                                         0)
        self.filtered_image=gray

    def highcontrast(self):
        img = self.original_image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        xp = [0, 64, 112, 128, 144, 192, 255]
        fp = [0, 16, 64, 128, 192, 240, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        img = cv2.LUT(gray, table)
        self.filtered_image = img

    def negative_button_released(self, event):
        self.negative()
        self.show_image()

    def black_white_released(self, event):
        self.black_white()
        self.show_image()

    def sepia_button_released(self, event):
        self.sepia()
        self.show_image()

    def emboss_button_released(self, event):
        self.emboss()
        self.show_image()

    def gaussian_blur_button_released(self, event):
        self.gaussian_blur()
        self.show_image()

    def median_blur_button_released(self, event):
        self.gaussian_blur()
        self.show_image()

    def apply_button_released(self, event):
        self.master.processed_image = self.filtered_image
        self.show_image()
        self.close()

    def cancel_button_released(self, event):
        self.master.image_viewer.show_image()
        self.close()

    def show_image(self):
        self.master.image_viewer.show_image(img=self.filtered_image)

    def negative(self):
        self.filtered_image = cv2.bitwise_not(self.original_image)

    def black_white(self):
        self.filtered_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.filtered_image = cv2.cvtColor(self.filtered_image, cv2.COLOR_GRAY2BGR)

    def sepia(self):
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])

        self.filtered_image = cv2.filter2D(self.original_image, -1, kernel)

    def emboss(self):
        kernel = np.array([[0, -1, -1],
                           [1, 0, -1],
                           [1, 1, 0]])

        self.filtered_image = cv2.filter2D(self.original_image, -1, kernel)

    def gaussian_blur(self):
        self.filtered_image = cv2.GaussianBlur(self.original_image, (41, 41), 0)

    def median_blur(self):
        self.filtered_image = cv2.medianBlur(self.original_image, 41)

    def close(self):
        self.destroy()
