'''
The `dynamicdl.processing` module handles file processing, including annotation files and
image files. These are to be used in describing DynamicDL dataset formats as values following
a File key indicator.

Classes:

* JSONFile

* TXTFile

* XMLFile

* YAMLFile

* ImageEntry

* SegmentationImage

'''
from .jsonfile import JSONFile
from .txtfile import TXTFile
from .xmlfile import XMLFile
from .yamlfile import YAMLFile
from .images import ImageEntry, SegmentationImage
