

class FlowGenerator:
    def __init__(self, input_images, output_dir, flow_warper):
        self.imgs = input_images
        self.output_dir = output_dir
        self.warper = flow_warper

    def gen(self):
        for i in range(len(self.imgs)):
            img1 = self.imgs[i]
            img2 = self.imgs[i + 1]
            print(img1, img2)
            img_w, flow_w = self.warper.warp(img1, img2)
