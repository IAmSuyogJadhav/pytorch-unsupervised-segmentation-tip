#from __future__ import print_function
# import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import cv2
import numpy as np
import torch.nn.init

use_cuda = torch.cuda.is_available()

class _Args(dict):
    """
    Converts the dictionary keys into attributes for easy access.
    """
    def __init__(self, *args, **kwargs):
        super(_Args, self).__init__(*args, **kwargs)
        self.__dict__ = self

# CNN model
class MyNet(nn.Module):
    def __init__(self, args, input_dim):
        super(MyNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


import cv2
import numpy as np

class Scribbler(object):
    def __init__(self, name, img):
        """
        Allows the user to draw a scribble and use for guiding the model.

        Parameters:
        -----------

        `name`: String, Required
            Name to give to the OpenCV window. Needs to be unique.
        `img`: NumPy array, Required
            The image to draw on, in OpenCV BGR format.
        """
        # For keeping track of the drawing
        self.last_x, self.last_y = -1, -1
        self.drawing = False

        # Make sure the image is 3-channeled (for visualization)
        if len(img.shape) == 3:
            if img.shape[2] == 3:  # Already 3 channels
                self.img = img.copy()  # The image to draw on
            else:
                self.img = np.repeat(img, 3, axis=2)
        else:  # Single channel, B&W image
            self.img = np.repeat(img[..., None], 3, axis=2)

        # Save the mask separately (2D)
        self.mask = np.full_like(img[..., 0], fill_value=255)  # Mask only, without the image
        
        # Coloring-related Stuff
        rng = np.random.RandomState(0)  # Get a deterministic generator
        self.colors = rng.randint(low=0, high=255, size=(255, 3))  # A list of 255 colors
        self.current_color = tuple([int(c) for c in self.colors[0]])  # Current color (start with 0th color)
        self.current_color_mask = 0  # Only color indices need to be saved in the mask
        self.thickness = 3  # In pixels

        # Create the OpenCV window
        self.name = name
        cv2.namedWindow(self.name, cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowTitle(self.name, '(Press [Esc] when done) ' + self.name)
        cv2.setMouseCallback(self.name, self.draw)  # Mouse callback for drawing
        cv2.createTrackbar("Segmentation Label", self.name, 0, 255, self.change_current_color)  # Change color
        cv2.createTrackbar("Thickness", self.name, 1, 20, self.change_current_thickness)  # Change thickness
        cv2.setTrackbarPos("Thickness", self.name, self.thickness)  # Set default value of thickness

    def change_current_color(self, new_idx):
        self.current_color = tuple([int(c) for c in self.colors[new_idx]])  # Required for OpenCV
        self.current_color_mask = new_idx

    def change_current_thickness(self, new_val):
        if new_val != 0:  # Make sure thickness value is valid
            self.thickness = new_val
        else:
            self.thickness = 1

    def draw(self, event, x, y, flags, param):
        """
        The main drawing function.
        """

        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y)
            self.drawing = True
            self.last_x, self.last_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                # Draw on the mask and also on the image (for visualization only, original image is unaltered)
                cv2.line(self.img, (self.last_x, self.last_y), (x, y), self.current_color, self.thickness)
                cv2.line(self.mask, (self.last_x, self.last_y), (x, y), self.current_color_mask, self.thickness)
                self.last_x, self.last_y = x, y
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.last_x != -1:  # Sanity check
                # Draw on the mask and also on the image (for visualization only, original image is unaltered)
                cv2.line(self.img, (self.last_x, self.last_y), (x, y), self.current_color, self.thickness)
                cv2.line(self.mask, (self.last_x, self.last_y), (x, y), self.current_color_mask, self.thickness)
                self.last_x, self.last_y = -1, -1

    def get_mask(self):
        """
        Launches an interactive OpenCV window for the user to draw a scribble in. 
        This scribble can then used by the model.

        Additional features:
            Segmentation label slider - Allows for drawing multiple segmentation labels.
            Thickness slider - Allows drawing with lines of different thicknesses (in pixels)

        Returns:
            `mask`: NumPy array
                The drawn scribble. Returned as a 2D numpy array of same height x width as the
                original image.

        """
        while(True):
            cv2.imshow(self.name, self.img)
            k=cv2.waitKey(1)&0xFF
            if k==27:
                break
            elif cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()

        return self.mask


class deep_unsupervised_segmentation(object):
    def __init__(self):
        pass
    def run(self, img, args, name):
        """
        Parameters:
        ----------

        `img`: np.array, Required
            The image in OpenCV BGR format.
        `args`: dict, Required
            The dictionary containing all the parameters needed for segmentation.
        `name`: String, Required
            Used for distinguising OpenCV windows created by this instance from the rest.

        Returns
        -------

        `im_target_concat`: np.array
            The output segmentation. Height and width is same as the input `img`.
            The image is single-channel with different segmentation labels identified
            by different pixel values.

        """
        args = _Args(**args)  # Turns keys into attributes

        # load image
        data = torch.from_numpy( np.array([img.transpose( (2, 0, 1) ).astype('float32')/255.]) )
        if use_cuda:
            data = data.cuda()
        data = Variable(data)

        # load scribble
        if args.scribble:
            ui = Scribbler(f'Scribble UI: {name}', img)  # Initialize the drawing UI
            mask = ui.get_mask()  # Get the mask from user
            mask = mask.reshape(-1)
            mask_inds = np.unique(mask)
            mask_inds = np.delete( mask_inds, np.argwhere(mask_inds==255) )
            inds_sim = torch.from_numpy( np.where( mask == 255 )[ 0 ] )
            inds_scr = torch.from_numpy( np.where( mask != 255 )[ 0 ] )
            target_scr = torch.from_numpy( mask.astype(np.int) )
            if use_cuda:
                inds_sim = inds_sim.cuda()
                inds_scr = inds_scr.cuda()
                target_scr = target_scr.cuda()
            target_scr = Variable( target_scr )
            # set minLabels
            args.minLabels = len(mask_inds)

        # train
        model = MyNet(args, data.size(1))
        if use_cuda:
            model.cuda()
        model.train()

        # similarity loss definition
        loss_fn = torch.nn.CrossEntropyLoss()

        # scribble loss definition
        loss_fn_scr = torch.nn.CrossEntropyLoss()

        # continuity loss definition
        loss_hpy = torch.nn.L1Loss(size_average = True)
        loss_hpz = torch.nn.L1Loss(size_average = True)

        HPy_target = torch.zeros(img.shape[0]-1, img.shape[1], args.nChannel)
        HPz_target = torch.zeros(img.shape[0], img.shape[1]-1, args.nChannel)
        if use_cuda:
            HPy_target = HPy_target.cuda()
            HPz_target = HPz_target.cuda()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        label_colours = np.random.randint(255,size=(args.nChannel,3))

        for batch_idx in range(args.maxIter):
            # forwarding
            optimizer.zero_grad()
            output = model( data )[ 0 ]
            output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )

            outputHP = output.reshape( (img.shape[0], img.shape[1], args.nChannel) )
            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            lhpy = loss_hpy(HPy,HPy_target)
            lhpz = loss_hpz(HPz,HPz_target)

            ignore, target = torch.max( output, 1 )
            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))
            if args.visualize:

                im_target_rgb = np.array([label_colours[ c % args.nChannel ] for c in im_target])
                im_target_rgb = im_target_rgb.reshape( img.shape ).astype( np.uint8 )
                cv2.imshow( f"Output: {name}", im_target_rgb )
                cv2.waitKey(10)

            # loss 
            if args.scribble:
                loss = args.stepsize_sim * loss_fn(output[ inds_sim ], target[ inds_sim ]) + args.stepsize_scr * loss_fn_scr(output[ inds_scr ], target_scr[ inds_scr ]) + args.stepsize_con * (lhpy + lhpz)
            else:
                loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)
                
            loss.backward()
            optimizer.step()

            print (batch_idx, '/', args.maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

            if nLabels <= args.minLabels:
                print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
                break

        # Return the output segmentation
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
        _, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()

        # Get Grayscale colors instead of RGB to make it easier to separate individual segmentations later
        label_colours = np.random.randint(255, size=(args.nChannel, 1))
        label_colours = np.repeat(label_colours, 3, axis=1)
        
        im_target_concat = np.array([label_colours[ c % args.nChannel ] for c in im_target])
        im_target_concat = im_target_concat.reshape( img.shape ).astype( np.uint8 )[..., 0] # Choosing any one channel since they are repetitive

        if args.visualize:
            cv2.destroyWindow(f"Output: {name}")  # Cleanup stray OpenCV window

        return im_target_concat  # Returning a single image back containing all the masks

