#!/usr/bin/python
# pyfacedetect - Thin wrapper around OpenCV library for detection
# and marking faces in images
# Copyright (C) 2010  Dejan Noveski <dr.mote@gmail.com>
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
pydetectfaces - Thin wrapper around OpenCV (http://opencv.willowgarage.com)
for detection and marking faces in images

The module can be used as API or in console for testing purposes. For usage
and options:
    $python pyfacedetect.py -h

The module consists of 2 classes:
    OcvDetector that wraps OpenCV Api for face detection and can be used bare
    without the helper methods for loading images, marking faces, or getting
    human friendly output

    FaceDetect - full feature class that extends OcvDetector and adds helper
    methods for loading/marking/outputing/saving images and dumping faces in
    json.

You can enable scaning for profiles for quantity of faces by setting 
SCAN_FOR_PROFILES to True. That, however will give some overlapping rectangles.

Try and tweak MIN_FACE_SIZE, HAAR_SCALE and MIN_NEIGHBORS so you can change the
accuracy of the detection. For more info, read 
http://opencv.willowgarage.com/documentation/python/objdetect_cascade_classification.html

For speed, all images above 1000width or 1000h are scaled to 1000w or 1000h max.

"""

import cv
from json import dumps
import argparse

MIN_FACE_SIZE = (20, 20) #minimal face size
HAAR_SCALE = 1.1
MIN_NEIGHBORS = 3
SCAN_FOR_PROFILES = False #Make this True to scan for profile faces also 


class OcvDetector(object):
    """ Wraper class for OpenCV. Manages the image preparation and
        face detection. """

    def load_image(self, image):
        """ Sets self.image to the frame that will be analyzed for
            faces. Based on the frame dimensions, calculates the
            scaling. For speed/accuracy, examined image will not 
            exceed 1000px height or width.

            Positional arguments:
                image -- cv.IplImage that will be stored for analyzing
        """

        self.image = image
        if image.width > 1000 or image.height>1000:
            if image.width > image.height:
                self.image_scale = image.width/1000
            else:
                self.image_scale = image.height/1000
        else:
            self.image_scale = 1
                

    def _prepare_image(self):
        """ Helper method that scales the image, converts it to greyscale
            and equalizes the histograms. This method provides significant
            speed-up of the detection process """
        
        #Alocate the gray copies and the scaled copies
        gray_copy = cv.CreateImage((self.image.width, self.image.height),
                                        8, 1)
        scaled_image = cv.CreateImage(
                            (cv.Round(self.image.width / self.image_scale),
			    cv.Round(self.image.height / self.image_scale)), 8, 1)
        
        #Convert the image to Grayscale and scale it by factor self.image_scale
        #from load_image method
        cv.CvtColor(self.image, gray_copy, cv.CV_BGR2GRAY)

        cv.Resize(gray_copy, scaled_image, cv.CV_INTER_LINEAR)
        cv.EqualizeHist(scaled_image, scaled_image)

        return scaled_image

    def detect_faces(self, include_profile_faces = SCAN_FOR_PROFILES):
        """ Detects the faces in self.image. For profile detection, turn
            SCAN_FOR_PROFILES to True.

            returns list of face rectangle dicts
        """

        self._faces = []
        scaled_image = self._prepare_image()

        #The cascade is loaded here. Change the path/filename to custom or other
        #cascade for experimenting
        cascade = cv.Load('haarcascade_frontalface_alt2.xml')

        #The magic method
        faces = cv.HaarDetectObjects(scaled_image, cascade, 
                                       cv.CreateMemStorage(0),
                                       HAAR_SCALE, MIN_NEIGHBORS,
                                       0,
                                       MIN_FACE_SIZE)

        #If the flag for profiles is set to True
        if include_profile_faces:
            cascade = cv.Load('haarcascade_profileface.xml')
            faces.extend(cv.HaarDetectObjects(scaled_image, cascade,
                                       cv.CreateMemStorage(0),
                                       HAAR_SCALE, MIN_NEIGHBORS, 
                                       0,
                                       MIN_FACE_SIZE))
        
        if faces:
            for ((x, y, w, h), n) in faces:
                self._faces.append({'x': x * self.image_scale, 'y': y * self.image_scale, 
                    'width': w * self.image_scale, 'height': h * self.image_scale, 'neighbors': n})

        return self._faces


class FaceDetect(OcvDetector):
    """ FaceDetect adds utility methods for loading images from different
        sources, saving output to file, dumping face rectangles as json,
        and overlaying the image with the face rectangles """

    def __init__(self):
        super(OcvDetector, self).__init__()
        self.is_overlayed = False #We have to store this value so we dont
        #save empty image

    def image_from_file(self, file_path):
        """ Loads the detector with image from file. All major formats
            supported.

            Positional arguments:
                file_path -- string The file path
        """

        self.load_image(cv.LoadImage(file_path))

    def image_from_input(self, input_id):
        """ Loads the detector with a frame from a capture device 
        (webcam/tv-card). Usualy input ID is 0 for webcams in Linux/OSX if it is
        default input. """

        capture = cv.CaptureFromCAM(input_id)
        if capture:
            frame = cv.QueryFrame(capture)#query the capture device for a frame
            self.load_image(frame)
            #We must hold a reference to the capture device. GC colects it before
            #it queries a frame and segfaults happen
            self.capture = capture

    def overlay_image(self, rgb_border = (255, 0, 0,), width = 2):
        """ Overlays IplImage with rectangles around the detected faces.
            The stored image will be changed (drawn) and we can save it, or
            show it.

            Keyword arguments:

            rgb_border -- tuple Color of the rectangle's border, defaults to red
            width -- int Width of the rectangle's border
        """

        if not self._faces:
            self.detect_faces()

        if self._faces:
            for face in self._faces:
                pt1 = (int(face['x']), int(face['y']))
                pt2 = (int(face['x'] + face['width']), int(face['y'] + face['height']))
                cv.Rectangle(self.image, pt1, pt2, cv.RGB(*rgb_border), width, 8, 0)
        self.is_overlayed = True
        return self.image

    def save_image(self, filename):
        """ Saves the overlayed image to a file.
            
            Positional arguments:
            filename -- string The file name
        """

        if not self.is_overlayed:
            self.overlay_image()

        return cv.SaveImage(filename, self.image)

    def show_image(self):
        """ Shows the image in OpenCV's HIGHGUI window. For testing only. """

        cv.NamedWindow('Output', 1)
        cv.ShowImage('Output', self.image)
        cv.WaitKey(100000)

    def to_json(self):
        """ Dumps the face rectangles to json string. Can be used for HTML/JS
            auto face taggers in web services etc. """

        return dumps(self._faces)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'pyfacedetect',
        description = """Detects faces in images or video inputs. 
        Outputs to image or window""",
        add_help = True)
    
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument('-i', '--input', action = 'store', type = int, 
            help = 'Video Input ID (usualy 0)')
    group.add_argument('-f', '--file', action = 'store', 
            help = 'File path')
    parser.add_argument('-s', '--save', action = 'store',
            help = 'File name to save overlayed image')
    parser.add_argument('-o', '--out', action = 'store_true',
            help = 'Output overlayed image to window')
    parser.add_argument('-j', '--json', action = 'store_true',
            help = 'Print Face rectangles json to console')
    args = parser.parse_args()
    if args.input >= 0 or args.file:
        detector = FaceDetect()
        if args.input >= 0:
            detector.image_from_input(args.input)
        else:
            detector.image_from_file(args.file)

        detector.detect_faces()
        detector.overlay_image()
        if args.json:
            print detector.to_json()
        if args.save:
            detector.save_image(args.save)
        if args.out:
            detector.show_image()

