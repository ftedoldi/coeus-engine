from asyncio import streams
from itertools import count
import math
from statistics import median
import time
from tkinter import (S, Button, Entry, ttk,Tk,PhotoImage,Canvas, filedialog, colorchooser,RIDGE,
                     GROOVE,ROUND,Scale,HORIZONTAL)
import tkinter
import cv2
from PIL import ImageTk, Image
import numpy as np
import numba
from numba import cuda
import cv2

class FrontEnd:
    def __init__(self, master):
        self.master = master
        self.menu_initialisation()
        
    def menu_initialisation(self):
        self.master.geometry('750x630+250+10')
        self.master.title('Photo Editing App with Python (Tkinter and OpenCV)')
        
        self.frame_header = ttk.Frame(self.master)
        self.frame_header.pack()
        self.d = dict()
        
        self.logo = PhotoImage(file='./Icons/logo.png').subsample(2, 2)
        ttk.Label(self.frame_header, image=self.logo).grid(
            row=0, column=0, rowspan=2)
        ttk.Label(self.frame_header, text='PhotoHub').grid(
            row=0, column=2, columnspan=1)
        ttk.Label(self.frame_header, text='An Image Editor Just For You!').grid(
            row=1, column=1, columnspan=3)
        
        
        self.frame_menu = ttk.Frame(self.master)
        self.frame_menu.pack()
        self.frame_menu.config(relief=RIDGE, padding=(50, 15))
        
        ttk.Button(
            self.frame_menu, text="Upload An Image", command=self.upload_action).grid(
            row=0, column=0, columnspan=2, padx=5, pady=5, sticky='sw')

        ttk.Button(
            self.frame_menu, text="Box Blur An Image", command=self.boxBlurGPU_action).grid(
            row=1, column=0, columnspan=2, padx=5, pady=5, sticky='sw')

        ttk.Button(
            self.frame_menu, text="Gaussian Blur An Image", command=self.gaussianBlurGPU_action).grid(
            row=2, column=0, columnspan=2, padx=5, pady=5, sticky='sw')

        ttk.Button(
            self.frame_menu, text="Median Blur An Image", command=self.medianBlurGPU_action).grid(
            row=3, column=0, columnspan=2, padx=5, pady=5, sticky='sw')

        # ttk.Button(
        #     self.frame_menu, text="Crop Image", command=self.crop_action).grid(
        #     row=2, column=0, columnspan=2, padx=5, pady=5, sticky='sw')

        # ttk.Button(
        #     self.frame_menu, text="Add Text", command=self.text_action_1).grid(
        #     row=3, column=0, columnspan=2, padx=5, pady=5, sticky='sw')

        # ttk.Button(
        #     self.frame_menu, text="Draw Over Image", command=self.draw_action).grid(
        #     row=4, column=0, columnspan=2, padx=5, pady=5, sticky='sw')

        # ttk.Button(
        #     self.frame_menu, text="Apply Filters", command=self.filter_action).grid(
        #     row=5, column=0, columnspan=2,  padx=5, pady=5, sticky='sw')

        ttk.Button(
            self.frame_menu, text="Blur/Smoothening", command=self.blur_action).grid(
            row=6, column=0, columnspan=2, padx=5, pady=5, sticky='sw')

        # ttk.Button(
        #     self.frame_menu, text="Adjust Levels", command=self.adjust_action).grid(
        #     row=7, column=0, columnspan=2, padx=5, pady=5, sticky='sw')

        # ttk.Button(
        #     self.frame_menu, text="Rotate", command=self.rotate_action).grid(
        #     row=8, column=0, columnspan=2, padx=5, pady=5, sticky='sw')

        # ttk.Button(
        #     self.frame_menu, text="Flip", command=self.flip_action).grid(
        #     row=9, column=0, columnspan=2, padx=5, pady=5, sticky='sw')

        ttk.Button(
            self.frame_menu, text="Save As", command=self.save_action).grid(
            row=10, column=0, columnspan=2, padx=5, pady=5, sticky='sw')
        
        self.canvas = Canvas(self.frame_menu, bg="gray", width=300, height=400)
        self.canvas.grid(row=0, column=3, rowspan=10)
        
        self.side_frame = ttk.Frame(self.frame_menu)
        self.side_frame.grid(row=0, column=4, rowspan=10)
        self.side_frame.config(relief=GROOVE, padding=(50,15))
        
        self.apply_and_cancel = ttk.Frame(self.master)
        self.apply_and_cancel.pack()
        self.apply = ttk.Button(self.apply_and_cancel, text="Apply", command=self.apply_action).grid(
            row=0, column=0, columnspan=1, padx=5, pady=5, sticky='sw')

        ttk.Button(
            self.apply_and_cancel, text="Cancel", command=self.cancel_action).grid(
                row=0, column=1, columnspan=1,padx=5, pady=5, sticky='sw')

        ttk.Button(
            self.apply_and_cancel, text="Revert All Changes", command=self.revert_action).grid(
                row=0, column=2, columnspan=1,padx=5, pady=5, sticky='sw')
        
    def upload_action(self):
        self.canvas.delete("all")
        self.filename = filedialog.askopenfilename()
        self.original_image = cv2.imread(self.filename)

        self.edited_image = cv2.imread(self.filename)
        self.filtered_image = cv2.imread(self.filename)
        self.display_image(self.edited_image)
    
    ###################################################### MY FUNCTIONS START ############################################################################    
    def boxBlurGPU_action(self):
        self.refresh_side_frame()
        
        ttk.Label(
            self.side_frame, text="Blur Intensity").grid(
            row=0, column=2, padx=5, sticky='sw')
            
        # We do that cos transparency is a bit shit
        self.img = cv2.imread(self.filename, cv2.IMREAD_UNCHANGED)
        if self.img.shape[2] == 4:     # we have an alpha channel
          a1 = ~self.img[:,:,3]        # extract and invert that alpha
          self.img = cv2.add(cv2.merge([a1,a1,a1,a1]), self.img)   # add up values (with clipping)
          self.img = cv2.cvtColor(self.img, cv2.COLOR_RGBA2RGB)    # strip alpha channel
          
        def sliderChanged(event):
            @cuda.jit
            def imageProcessing(img, x, blurIntensity):
                pos = cuda.grid(2)
                shape = img.shape
                
                if pos[0] < shape[0] and pos[1] < shape[1]:
                    firstElementPositionX = pos[0]
                    firstElementPositionY = pos[1]
                    rowElements = 0
                    c = 0
                    for i in range(blurIntensity):
                        if pos[0] - (i + 1) >= 0:
                            firstElementPositionX = pos[0] - (i + 1)
                            firstElementPositionY = pos[1] - (i + 1)
                            rowElements = 3 + (2 * c)
                            c += 1
                            
                    div = 0
                    r = 0
                    g = 0
                    b = 0
                    
                    for i in range(rowElements):
                        for j in range(rowElements):
                            if firstElementPositionX + i < shape[0] and firstElementPositionY + j < shape[1]:
                                r += img[firstElementPositionX + i][firstElementPositionY + j][0]
                                g += img[firstElementPositionX + i][firstElementPositionY + j][1]
                                b += img[firstElementPositionX + i][firstElementPositionY + j][2]
                                div += 1
                                
                    if div > 0:
                        img[pos[0]][pos[1]][0] = r // div
                        img[pos[0]][pos[1]][1] = g // div
                        img[pos[0]][pos[1]][2] = b // div
                                
                #     mult = 0
                #     div = 0

                #     for i in range(1, blurIntensity):
                #         newX = x * (i * 3)
                #         if pos - newX >= 0:
                #             mult += img[pos - newX]
                #             div += 1
                #         if pos + newX < img.size:
                #             mult += img[pos + newX]
                #             div += 1
                #         if pos - (i * 3) >= 0:
                #             mult += img[pos - (i * 3)]
                #             div += 1
                #         if pos + (i * 3) < img.size:
                #             mult += img[pos + (i * 3)]
                #             div += 1

                #     if mult > 0:
                #         img[pos] = mult / div
                         
            def CPUFilter(img, x, blurIntensity):
                for pos in range(img.size):
                    mult = 0
                    div = 0
                    
                    for i in range(1, blurIntensity):
                        newX = x * i
                        if pos - newX >= 0:
                            mult += img[pos - newX]
                            div += 1
                        if pos - newX - 1 >= 0:
                            mult += img[pos - newX - 1]
                            div += 1
                        if pos - newX + 1 >= 0:
                            mult += img[pos - newX + 1]
                            div += 1
                        if pos + newX < img.size:
                            mult += img[pos + newX]
                            div += 1
                        if pos + newX + 1 < img.size:
                            mult += img[pos + newX + 1]
                            div += 1
                        if pos + newX - 1 < img.size:
                            mult += img[pos + newX - 1]
                            div += 1
                        if pos - i >= 0:
                            mult += img[pos - i]
                            div += 1
                        if pos + i < img.size:
                            mult += img[pos + i]
                            div += 1
                            
                    if mult > 0:
                        img[pos] = mult / div

            # with streams we are have a speedup of ~1.5
            original_shape = self.img.shape
            img = self.img
            
            sliderValue = intensitySlider.get()
            
            n = original_shape[0] * original_shape[1] * original_shape[2]
            chunks = math.ceil(img.size / n)
            threadsperblock = 32, 32
            blockspergrid = n // (threadsperblock[0] * threadsperblock[1]), n // (threadsperblock[0] * threadsperblock[1])
            
            stream = []
            splittedImage = []
            for i in range(chunks):
                start = i * n
                end = start + n
                splittedImage += [img[start:end]]
                stream += [cuda.stream()]
            for i in range(chunks):
                with stream[i].auto_synchronize():
                    d_img = cuda.to_device(splittedImage[i], stream=stream[i])
                    imageProcessing[blockspergrid, threadsperblock, stream[i]](d_img, original_shape[0], sliderValue)
                    d_img.copy_to_host(splittedImage[i], stream=stream[i])
            
            # seconds = time.time()
            # threadsperblock = 32
            # blockspergrid = (img.size + (threadsperblock - 1)) // threadsperblock
            # d1_img = cuda.to_device(img)
            # imageProcessing[blockspergrid, threadsperblock](d1_img, original_shape[0], intensitySlider.get())
            # d1_img.copy_to_host(img)
            # normalTime = time.time() - seconds
            # print(normalTime)

            # ~10 time slower
            # CPUFilter(img, original_shape[0], intensitySlider.get())
            
            # if streamTime - normalTime < 0:
            #     print("GAIN:")
            #     print(streamTime / normalTime)
            #     print("------------------------------")
            # else:
            #     print("NO GAIN")

            img = np.reshape(img, original_shape)
            self.display_image(image=img)
            self.filtered_image = img
        
        intensityValue = tkinter.IntVar()
        intensitySlider = Scale(
            self.side_frame, from_=0, to=256, orient=HORIZONTAL, variable=intensityValue, command=sliderChanged)
        intensitySlider.grid(row=1, column=2, padx=5,  sticky='sw')
    
    def gaussianBlurGPU_action(self):
        self.refresh_side_frame()
                    
        # We do that cos transparency is a bit shit
        self.img = cv2.imread(self.filename, cv2.IMREAD_UNCHANGED)
        if self.img.shape[2] == 4:     # we have an alpha channel
          a1 = ~self.img[:,:,3]        # extract and invert that alpha
          self.img = cv2.add(cv2.merge([a1,a1,a1,a1]), self.img)   # add up values (with clipping)
          self.img = cv2.cvtColor(self.img, cv2.COLOR_RGBA2RGB)    # strip alpha channel
          
        def sliderChanged(event):
            #TODO: Make it work with colors
            @cuda.jit
            def imageProcessing(img, blurIntensity, iterations):
                pos = cuda.grid(1)
                col = 0
                sum = 0
                if pos < img.size:
                    for i in range(iterations):
                        offset = ((i / (iterations - 1)) - 0.5)
                        if pos + (i * 3) < img.size:
                            gaussian = (1 / (math.sqrt(2 * math.pi * blurIntensity * blurIntensity) )) * pow(math.e, -(pow(offset, 2) / (2 * pow(blurIntensity, 2))))
                            col += img[pos + (i * 3)] * gaussian
                            sum += gaussian
                    img[pos] = col / sum

            #TODO: with streams break data in chunks
            seconds = time.time()
            original_shape = self.img.shape
            img = self.img.flatten()

            # ~10 time slower
            # CPUFilter(img, original_shape[0], intensitySlider.get())
            d_img = cuda.to_device(img)
            threadsperblock = 32
            blockspergrid = (img.size + (threadsperblock - 1)) // threadsperblock
            blurStrength = intensitySlider.get() / 1280
            iterations = iterationsSlider.get()
            imageProcessing[blockspergrid, threadsperblock](d_img, blurStrength, iterations)
            img = d_img.copy_to_host()

            img = np.reshape(img, original_shape)
            self.display_image(image=img)
            print(time.time() - seconds)
        
        intensityValue = tkinter.DoubleVar()
        iterationValue = tkinter.IntVar()
        ttk.Label(
            self.side_frame, text="Blur Intensity").grid(
            row=0, column=2, padx=5, sticky='sw')
        intensitySlider = Scale(
            self.side_frame, from_=1, to=256, orient=HORIZONTAL, variable=intensityValue, command=sliderChanged)
        intensitySlider.grid(row=1, column=2, padx=5,  sticky='sw')
        ttk.Label(
            self.side_frame, text="Blur Iterations").grid(
            row=2, column=2, padx=5, sticky='sw')
        iterationsSlider = Scale(
            self.side_frame, from_=20, to=500, orient=HORIZONTAL, variable=iterationValue, command=sliderChanged)
        iterationsSlider.grid(row=3, column=2, padx=5,  sticky='sw')
    
    def medianBlurGPU_action(self):
        self.refresh_side_frame()
        
        ttk.Label(
            self.side_frame, text="Blur Intensity").grid(
            row=0, column=2, padx=5, sticky='sw')
            
        # We do that cos transparency is a bit shit
        self.img = cv2.imread(self.filename, cv2.IMREAD_UNCHANGED)
        if self.img.shape[2] == 4:     # we have an alpha channel
          a1 = ~self.img[:,:,3]        # extract and invert that alpha
          self.img = cv2.add(cv2.merge([a1,a1,a1,a1]), self.img)   # add up values (with clipping)
          self.img = cv2.cvtColor(self.img, cv2.COLOR_RGBA2RGB)    # strip alpha channel
          
        def sliderChanged():
            @cuda.jit
            def imageProcessing(img):
                pos = cuda.grid(2)
                shape = img.shape
                if pos[0] < shape[0] and pos[1] < shape[1]:
                    # rowSize = 3 + (2 * blurIntensity)
                    # size = rowSize * rowSize
                    median = cuda.local.array((25 * 3, ), np.int16)
                    
                    firstElementPositionX = pos[0]
                    firstElementPositionY = pos[1]
                    rowElements = 0
                    c = 0
                    for i in range(2):
                        if pos[0] - (i + 1) >= 0:
                            firstElementPositionX = pos[0] - (i + 1)
                            firstElementPositionY = pos[1] - (i + 1)
                            rowElements = 3 + (2 * c)
                            c += 1

                    counter = 0
                    for i in range(rowElements):
                        for j in range(rowElements):
                            if firstElementPositionX + i < shape[0] and firstElementPositionY + j < shape[1]:
                                median[counter] = img[firstElementPositionX + i][firstElementPositionY + j][0]
                                median[counter + 1] = img[firstElementPositionX + i][firstElementPositionY + j][1]
                                median[counter + 2] = img[firstElementPositionX + i][firstElementPositionY + j][2]
                            else:
                                median[counter] = -1
                                median[counter + 1] = -1
                                median[counter + 2] = -1
                            counter += 3              
                        
                    offset = 0
                    for i in range(0, len(median), 3):
                        if median[i] == -1:
                            offset += 3
                        
                    for i in range(0, len(median), 3):
                        for j in range(i + 3, len(median), 3):
                            first = median[i] + median[i + 1] + median[i + 2]
                            second = median[j] + median[j + 1] + median[j + 2]
                            if first > second:
                                median[i], median[j] = median[j], median[i]
                                median[i + 1], median[j + 1] = median[j + 1], median[i + 1]
                                median[i + 2], median[j + 2] = median[j + 2], median[i + 2]

                    mid = ((len(median) - offset) // 3) // 2
                    normalMid = 0
                    for i in range(0, mid, 3):
                        normalMid = i
                        
                    if median[normalMid] != -1 and median[normalMid + 1] != -1 and median[normalMid + 2] != -1:
                        if median[normalMid] != 0 and median[normalMid + 1] != 0 and median[normalMid + 2] != 0:
                            img[pos[0]][pos[1]][0] = median[normalMid]
                            img[pos[0]][pos[1]][1] = median[normalMid + 1]
                            img[pos[0]][pos[1]][2] = median[normalMid + 2]
                            
            def CPUFilter(img, x, blurIntensity):
                for pos in range(img.size):
                    median = []
                    
                    for i in range(1, blurIntensity):
                        newX = x * i
                        if pos - newX >= 0:
                            median += img[pos - newX]
                        if pos - newX - 1 >= 0:
                            median += img[pos - newX - 1]
                        if pos - newX + 1 >= 0:
                            median += img[pos - newX + 1]
                        if pos + newX < img.size:
                            median += img[pos + newX]
                        if pos + newX + 1 < img.size:
                            median += img[pos + newX + 1]
                        if pos + newX - 1 < img.size:
                            median += img[pos + newX - 1]
                        if pos - i >= 0:
                            median += img[pos - i]
                        if pos + i < img.size:
                            median += img[pos + i]
                            
                    img[pos] = median[len(median) // 2]

            #TODO: with streams break data in chunks
            #seconds = time.time()
            img = self.img

            # ~10 time slower
            # CPUFilter(img, original_shape[0], intensitySlider.get())
            # seconds = time.time()
            # stream = []
            # n = original_shape[0] * original_shape[1] * original_shape[2]
            # chunks = math.ceil(img.size / n)
            # threadsperblock = 32, 32
            # blockspergrid = n // (threadsperblock[0] * 2), n // (threadsperblock[1] * 2)
            # sliderValue = 1
            # splittedImage = []
            # for i in range(chunks):
            #     start = i * n
            #     end = start + n
            #     splittedImage += [img[start:end]]
            #     stream += [cuda.stream()]
            # for i in range(chunks):
            #     d_img = cuda.to_device(splittedImage[i], stream=stream[i])
            #     imageProcessing[blockspergrid, threadsperblock, stream[i]](d_img)
            #     d_img.copy_to_host(splittedImage[i], stream=stream[i])
            seconds = time.time()
            threadsperblock = 32, 32
            tpb = threadsperblock[0] * threadsperblock[1]
            blockspergrid = img.size // (tpb * 2), img.size // (tpb * 2)
            stream = cuda.stream()
            d_img = cuda.to_device(img, stream=stream)
            imageProcessing[blockspergrid, threadsperblock, stream](d_img)
            d_img.copy_to_host(img, stream=stream)

            # img = np.reshape(img, original_shape)
            self.display_image(image=img)
            self.filtered_image = img
            print(time.time() - seconds)
        
        # intensityValue = tkinter.IntVar()
        # intensityEntry = Entry(self.side_frame)
        intensityButton = Button(self.side_frame, text="Apply filter", command=sliderChanged)
        # intensityEntry.grid(row=1, column=2, padx=5,  sticky='sw')
        intensityButton.grid(row=2, column=2, padx=5,  sticky='sw')
        # intensitySlider = Scale(
        #     self.side_frame, from_=0, to=256, orient=HORIZONTAL, variable=intensityValue, command=sliderChanged)
        # intensitySlider.grid(row=1, column=2, padx=5,  sticky='sw')
    ###################################################### MY FUNCTIONS END ############################################################################
    
    def text_action_1(self):
        self.text_extracted = "hello"
        self.refresh_side_frame()
        ttk.Label(
            self.side_frame, text="Enter the text").grid(row=0, column=2, padx=5, pady=5, sticky='se')
        self.text_on_image = ttk.Entry(self.side_frame)
        self.text_on_image.grid(row=1, column=2, padx=5, sticky='se')
        ttk.Button(
            self.side_frame, text="Pick A Font Color", command=self.choose_color).grid(
            row=2, column=2, padx=5, pady=5, sticky='se')
        self.text_action()


    def crop_action(self):
        self.rectangle_id = 0
        # self.ratio = 0
        self.crop_start_x = 0
        self.crop_start_y = 0
        self.crop_end_x = 0
        self.crop_end_y = 0
        self.canvas.bind("<ButtonPress>", self.start_crop)
        self.canvas.bind("<B1-Motion>", self.crop)
        self.canvas.bind("<ButtonRelease>", self.end_crop)

    def start_crop(self, event):
        self.crop_start_x = event.x
        self.crop_start_y = event.y

    def crop(self, event):
        if self.rectangle_id:
            self.canvas.delete(self.rectangle_id)

        self.crop_end_x = event.x
        self.crop_end_y = event.y

        self.rectangle_id = self.canvas.create_rectangle(self.crop_start_x, self.crop_start_y,
                                                         self.crop_end_x, self.crop_end_y, width=1)

    def end_crop(self, event):
        if self.crop_start_x <= self.crop_end_x and self.crop_start_y <= self.crop_end_y:
            start_x = int(self.crop_start_x * self.ratio)
            start_y = int(self.crop_start_y * self.ratio)
            end_x = int(self.crop_end_x * self.ratio)
            end_y = int(self.crop_end_y * self.ratio)
        elif self.crop_start_x > self.crop_end_x and self.crop_start_y <= self.crop_end_y:
            start_x = int(self.crop_end_x * self.ratio)
            start_y = int(self.crop_start_y * self.ratio)
            end_x = int(self.crop_start_x * self.ratio)
            end_y = int(self.crop_end_y * self.ratio)
        elif self.crop_start_x <= self.crop_end_x and self.crop_start_y > self.crop_end_y:
            start_x = int(self.crop_start_x * self.ratio)
            start_y = int(self.crop_end_y * self.ratio)
            end_x = int(self.crop_end_x * self.ratio)
            end_y = int(self.crop_start_y * self.ratio)
        else:
            start_x = int(self.crop_end_x * self.ratio)
            start_y = int(self.crop_end_y * self.ratio)
            end_x = int(self.crop_start_x * self.ratio)
            end_y = int(self.crop_start_y * self.ratio)

        x = slice(start_x, end_x, 1)
        y = slice(start_y, end_y, 1)

        self.filtered_image = self.edited_image[y, x]
        self.display_image(self.filtered_image)

    def text_action(self):
        self.rectangle_id = 0
        # self.ratio = 0
        self.crop_start_x = 0
        self.crop_start_y = 0
        self.crop_end_x = 0
        self.crop_end_y = 0
        self.canvas.bind("<ButtonPress>", self.start_crop)
        self.canvas.bind("<B1-Motion>", self.crop)
        self.canvas.bind("<ButtonRelease>", self.end_text_crop)

    def end_text_crop(self, event):
        if self.crop_start_x <= self.crop_end_x and self.crop_start_y <= self.crop_end_y:
            start_x = int(self.crop_start_x * self.ratio)
            start_y = int(self.crop_start_y * self.ratio)
            end_x = int(self.crop_end_x * self.ratio)
            end_y = int(self.crop_end_y * self.ratio)
        elif self.crop_start_x > self.crop_end_x and self.crop_start_y <= self.crop_end_y:
            start_x = int(self.crop_end_x * self.ratio)
            start_y = int(self.crop_start_y * self.ratio)
            end_x = int(self.crop_start_x * self.ratio)
            end_y = int(self.crop_end_y * self.ratio)
        elif self.crop_start_x <= self.crop_end_x and self.crop_start_y > self.crop_end_y:
            start_x = int(self.crop_start_x * self.ratio)
            start_y = int(self.crop_end_y * self.ratio)
            end_x = int(self.crop_end_x * self.ratio)
            end_y = int(self.crop_start_y * self.ratio)
        else:
            start_x = int(self.crop_end_x * self.ratio)
            start_y = int(self.crop_end_y * self.ratio)
            end_x = int(self.crop_start_x * self.ratio)
            end_y = int(self.crop_start_y * self.ratio)

        if self.text_on_image.get():
            self.text_extracted = self.text_on_image.get()
        start_font = start_x, start_y
        r, g, b = tuple(map(int, self.color_code[0]))

        self.filtered_image = cv2.putText(
            self.edited_image, self.text_extracted, start_font, cv2.FONT_HERSHEY_SIMPLEX, 2, (b, g, r), 5)
        self.display_image(self.filtered_image)

    def draw_action(self):
        self.color_code = ((255, 0, 0), '#ff0000')
        self.refresh_side_frame()
        self.canvas.bind("<ButtonPress>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.draw_color_button = ttk.Button(
            self.side_frame, text="Pick A Color", command=self.choose_color)
        self.draw_color_button.grid(
            row=0, column=2, padx=5, pady=5, sticky='sw')

    def choose_color(self):
        self.color_code = colorchooser.askcolor(title="Choose color")

    def start_draw(self, event):
        self.x = event.x
        self.y = event.y
        self.draw_ids = []

    def draw(self, event):
        print(self.draw_ids)
        self.draw_ids.append(self.canvas.create_line(self.x, self.y, event.x, event.y, width=2,
                                                     fill=self.color_code[-1], capstyle=ROUND, smooth=True))

        cv2.line(self.filtered_image, (int(self.x * self.ratio), int(self.y * self.ratio)),
                 (int(event.x * self.ratio), int(event.y * self.ratio)),
                 (0, 0, 255), thickness=int(self.ratio * 2),
                 lineType=8)

        self.x = event.x
        self.y = event.y
    
    def refresh_side_frame(self):
        try:
            self.side_frame.grid_forget()
        except:
            pass

        self.canvas.unbind("<ButtonPress>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease>")
        self.display_image(self.edited_image)
        self.side_frame = ttk.Frame(self.frame_menu)
        self.side_frame.grid(row=0, column=4, rowspan=10)
        self.side_frame.config(relief=GROOVE, padding=(50, 15))

    def filter_action(self):
        self.refresh_side_frame()
        ttk.Button(
            self.side_frame, text="Negative", command=self.negative_action).grid(
            row=0, column=2, padx=5, pady=5, sticky='se')

        ttk.Button(
            self.side_frame, text="Black And white", command=self.bw_action).grid(
            row=1, column=2, padx=5, pady=5, sticky='se')

        ttk.Button(
            self.side_frame, text="Stylisation", command=self.stylisation_action).grid(
            row=2, column=2, padx=5, pady=5, sticky='se')

        ttk.Button(
            self.side_frame, text="Sketch Effect", command=self.sketch_action).grid(
            row=3, column=2, padx=5, pady=5, sticky='se')

        ttk.Button(
            self.side_frame, text="Emboss", command=self.emb_action).grid(
            row=4, column=2, padx=5, pady=5, sticky='se')

        ttk.Button(
            self.side_frame, text="Sepia", command=self.sepia_action).grid(
            row=5, column=2, padx=5, pady=5, sticky='se')

        ttk.Button(
            self.side_frame, text="Binary Thresholding", command=self.binary_threshold_action).grid(
            row=6, column=2, padx=5, pady=5, sticky='se')

        ttk.Button(
            self.side_frame, text="Erosion", command=self.erosion_action).grid(
            row=7, column=2, padx=5, pady=5, sticky='se')

        ttk.Button(
            self.side_frame, text="Dilation", command=self.dilation_action).grid(
            row=8, column=2, padx=5, pady=5, sticky='se')
    
    def blur_action(self):
        self.refresh_side_frame()

        ttk.Label(
            self.side_frame, text="Averaging Blur").grid(
            row=0, column=2, padx=5, sticky='sw')

        self.average_slider = Scale(
            self.side_frame, from_=0, to=256, orient=HORIZONTAL, command=self.averaging_action)
        self.average_slider.grid(row=1, column=2, padx=5,  sticky='sw')

        ttk.Label(
            self.side_frame, text="Gaussian Blur").grid(row=2, column=2, padx=5, sticky='sw')

        self.gaussian_slider = Scale(
            self.side_frame, from_=0, to=256, orient=HORIZONTAL, command=self.gaussian_action)
        self.gaussian_slider.grid(row=3, column=2, padx=5,  sticky='sw')

        ttk.Label(
            self.side_frame, text="Median Blur").grid(row=4, column=2, padx=5, sticky='sw')

        self.median_slider = Scale(
            self.side_frame, from_=0, to=256, orient=HORIZONTAL, command=self.median_action)
        self.median_slider.grid(row=5, column=2, padx=5,  sticky='sw')
    
    def rotate_action(self):
        self.refresh_side_frame()
        ttk.Button(
            self.side_frame, text="Rotate Left", command=self.rotate_left_action).grid(
            row=0, column=2, padx=5, pady=5, sticky='sw')

        ttk.Button(
            self.side_frame, text="Rotate Right", command=self.rotate_right_action).grid(
            row=1, column=2, padx=5, pady=5, sticky='sw')

    
    def flip_action(self):
        self.refresh_side_frame()
        ttk.Button(
            self.side_frame, text="Vertical Flip", command=self.vertical_action).grid(
            row=0, column=2, padx=5, pady=5, sticky='se')

        ttk.Button(
            self.side_frame, text="Horizontal Flip", command=self.horizontal_action).grid(
            row=1, column=2, padx=5, pady=5, sticky='se')
    
    def adjust_action(self):
        self.refresh_side_frame()
        ttk.Label(
            self.side_frame, text="Brightness").grid(row=0, column=2, padx=5, pady=5, 
                                                     sticky='sw')

        self.brightness_slider = Scale(
            self.side_frame, from_=0, to_=2,  resolution=0.1, orient=HORIZONTAL, 
            command=self.brightness_action)
        self.brightness_slider.grid(row=1, column=2, padx=5,  sticky='sw')
        self.brightness_slider.set(1)

        ttk.Label(
            self.side_frame, text="Saturation").grid(row=2, column=2, padx=5, 
                                                     pady=5, sticky='sw')
        self.saturation_slider = Scale(
            self.side_frame, from_=-200, to=200, resolution=0.5, orient=HORIZONTAL, 
            command=self.saturation_action)
        self.saturation_slider.grid(row=3, column=2, padx=5,  sticky='sw')
        self.saturation_slider.set(0)
    
    def save_action(self):
        original_file_type = self.filename.split('.')[-1]
        filename = filedialog.asksaveasfilename()
        filename = filename + "." + original_file_type

        save_as_image = self.edited_image
        cv2.imwrite(filename, save_as_image)
        self.filename = filename
    
    def negative_action(self):
        self.filtered_image = cv2.bitwise_not(self.edited_image)
        self.display_image(self.filtered_image)

    def bw_action(self):
        self.filtered_image = cv2.cvtColor(
            self.edited_image, cv2.COLOR_BGR2GRAY)
        self.filtered_image = cv2.cvtColor(
            self.filtered_image, cv2.COLOR_GRAY2BGR)
        self.display_image(self.filtered_image)

    def stylisation_action(self):
        self.filtered_image = cv2.stylization(
            self.edited_image, sigma_s=150, sigma_r=0.25)
        self.display_image(self.filtered_image)

    def sketch_action(self):
        ret, self.filtered_image = cv2.pencilSketch(
            self.edited_image, sigma_s=60, sigma_r=0.5, shade_factor=0.02)
        self.display_image(self.filtered_image)

    def emb_action(self):
        kernel = np.array([[0, -1, -1],
                           [1, 0, -1],
                           [1, 1, 0]])
        self.filtered_image = cv2.filter2D(self.original_image, -1, kernel)
        self.display_image(self.filtered_image)

    def sepia_action(self):
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])

        self.filtered_image = cv2.filter2D(self.original_image, -1, kernel)
        self.display_image(self.filtered_image)

    def binary_threshold_action(self):
        ret, self.filtered_image = cv2.threshold(
            self.edited_image, 127, 255, cv2.THRESH_BINARY)
        self.display_image(self.filtered_image)

    def erosion_action(self):
        kernel = np.ones((5, 5), np.uint8)
        self.filtered_image = cv2.erode(
            self.edited_image, kernel, iterations=1)
        self.display_image(self.filtered_image)

    def dilation_action(self):
        kernel = np.ones((5, 5), np.uint8)
        self.filtered_image = cv2.dilate(
            self.edited_image, kernel, iterations=1)
        self.display_image(self.filtered_image)
    
    def averaging_action(self, value):
        value = int(value)
        if value % 2 == 0:
            value += 1
        self.filtered_image = cv2.blur(self.edited_image, (value, value))
        self.display_image(self.filtered_image)

    def gaussian_action(self, value):
        value = int(value)
        if value % 2 == 0:
            value += 1
        self.filtered_image = cv2.GaussianBlur(
            self.edited_image, (value, value), 0)
        self.display_image(self.filtered_image)

    def median_action(self, value):
        value = int(value)
        if value % 2 == 0:
            value += 1
        self.filtered_image = cv2.medianBlur(self.edited_image, value)
        self.display_image(self.filtered_image)
    
    def brightness_action(self, value):
        self.filtered_image = cv2.convertScaleAbs(
            self.filtered_image, alpha=self.brightness_slider.get())
        self.display_image(self.filtered_image)

    def saturation_action(self, event):
        self.filtered_image = cv2.convertScaleAbs(
            self.filtered_image, alpha=1, beta=self.saturation_slider.get())
        self.display_image(self.filtered_image)
    
    def rotate_left_action(self):
        self.filtered_image = cv2.rotate(
            self.filtered_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.display_image(self.filtered_image)

    def rotate_right_action(self):
        self.filtered_image = cv2.rotate(
            self.filtered_image, cv2.ROTATE_90_CLOCKWISE)
        self.display_image(self.filtered_image)

    def vertical_action(self):
        self.filtered_image = cv2.flip(self.filtered_image, 0)
        self.display_image(self.filtered_image)

    def horizontal_action(self):
        self.filtered_image = cv2.flip(self.filtered_image, 2)
        self.display_image(self.filtered_image)
    
    def apply_action(self):
        self.edited_image = self.filtered_image
        self.display_image(self.edited_image)

    def cancel_action(self):
        self.display_image(self.edited_image)

    def revert_action(self):
        self.edited_image = self.original_image.copy()
        self.display_image(self.original_image)
    
    def display_image(self, image=None):
        self.canvas.delete("all")
        if image is None:
            image = self.edited_image.copy()
        else:
            image = image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = image.shape
        ratio = height / width

        new_width = width
        new_height = height

        if height > 400 or width > 300:
            if ratio < 1:
                new_width = 300
                new_height = int(new_width * ratio)
            else:
                new_height = 400
                new_width = int(new_height * (width / height))

        self.ratio = height / new_height
        self.new_image = cv2.resize(image, (new_width, new_height))

        self.new_image = ImageTk.PhotoImage(
            Image.fromarray(self.new_image))

        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(
            new_width / 2, new_height / 2,  image=self.new_image)
    

mainWindow = Tk()
FrontEnd(mainWindow)
mainWindow.mainloop()

