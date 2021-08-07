#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tkinter import *
import tkinter.filedialog
import tkinter.messagebox
import tkinter.filedialog
import os
from PIL import Image, ImageTk
from feature.feature_extral_comp import FeatureExtAndComp
from feature.config import Config

class RetrieverGUI():
    def __init__(self, window):

        self.retriever = FeatureExtAndComp(Config.arch_name, Config.class_nums,
                                           Config.input_size, Config.batch_size,
                                           Config.feature_layer_name, Config.feature_index_in_module)

        self.parent = window
        self.parent.title("图像检索工具")

        #load constrast Image
        self.frame = Frame(self.parent)
        self.frame.grid(row=0, column = 0, sticky=W)

        self.contrast_label = Label(self.frame, text="ContrastImage:")
        self.contrast_label.grid(row=0, column=0, sticky=W)
        self.contrast_entry = Entry(self.frame)
        self.contrast_entry.grid(row=0, column=1, sticky=W)
        self.contrast_btn = Button(self.frame, text="Load", command=self.choose_contrast_file)
        self.contrast_btn.grid(row=0, column=2, sticky=W)

        self.contrast_panel = Canvas(self.frame)
        self.contrast_panel.grid(row=1, column=0, rowspan=2,columnspan=2,
                                 sticky=W+E+N+S, padx=5, pady=5)#合并两行，两列，居中，四周外延5个长度



        self.btn = Button(self.frame, text="Retriever", command=self.get_retriever_top)
        self.btn.grid(row=1, column=3, sticky=W + E)


        self.retrieved_label = Label(self.frame, text="RetrievedDir:")
        self.retrieved_label.grid(row=0, column=4, sticky=W)
        self.retrieved_entry = Entry(self.frame)
        self.retrieved_entry.grid(row=0, column=5, sticky=W)
        self.retrieved_btn = Button(self.frame, text="Load", command=self.choose_retrieved_dir)
        self.retrieved_btn.grid(row=0, column=6, sticky=W+E)


        self.result_panel1 = Canvas(self.frame)
        self.result_panel1.grid(row=1, column=4, rowspan=2,columnspan=2,
                                 sticky=W+E+N+S, padx=5, pady=5)#合并两行，两列，居中，四周外延5个长度



    def choose_retrieved_dir(self):  # 选择文件
        self.selectDirName = tkinter.filedialog.askdirectory(title='选择目录')
        self.retrieved_entry.insert(0, self.selectDirName.split('/')[-1])
        if not os.path.exists(self.selectDirName):
            tkinter.messagebox.askokcancel("Error!", message="The specified dir doesn't exist!")
            return
        print(self.selectDirName)



    def choose_contrast_file(self):  # 选择文件
        self.selectFileName = tkinter.filedialog.askopenfilename(title='选择文件')
        if not os.path.exists(self.selectFileName):
            tkinter.messagebox.askokcancel("Error!", message = "The specified image doesn't exist!")
            return

        file_name = self.selectFileName.split('/')[-1]
        self.contrast_entry.insert(0,file_name)
        self.img_png = Image.open(self.selectFileName)
        self.tkimg = ImageTk.PhotoImage(self.img_png)
        self.contrast_panel.config(width=self.tkimg.width(), height=self.tkimg.height())
        self.contrast_panel.create_image(0, 0, image=self.tkimg, anchor=NW)

    def get_retriever_top(self):
        result = self.retriever.get_topN(1, self.selectFileName, self.selectDirName)
        result_image= os.path.join(self.selectDirName, result[0])
        self.result_png = Image.open(result_image)
        self.tkresultimg = ImageTk.PhotoImage(self.result_png)
        self.result_panel1.config(width=self.tkresultimg.width(), height=self.tkresultimg.height())
        self.result_panel1.create_image(0, 0, image=self.tkresultimg, anchor=NW)


if __name__ == '__main__':
    window = Tk()
    retriever_gui = RetrieverGUI(window)
    window.resizable(width=True, height=True)
    window.mainloop()          #父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示
