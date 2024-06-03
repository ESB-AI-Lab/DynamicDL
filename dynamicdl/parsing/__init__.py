'''
The `dynamicdl.parsing` module handles the objects used for DynamicDL format creation. These objects
are to be used in the form when constructing the DynamicDL loader.

Classes:

* Static

* Generic

* Folder
   
* File
   
* ImageFile
   
* Alias

* Namespace

* GenericList

* SegmentationObject

* AmbiguousList

* ImpliedList

* Pairing

'''
from .static import Static
from .generic import Generic, Folder, File, ImageFile
from .alias import Alias
from .namespace import Namespace
from .genericlist import GenericList
from .segmentationobject import SegmentationObject
from .ambiguouslist import AmbiguousList
from .impliedlist import ImpliedList
from .pairing import Pairing
