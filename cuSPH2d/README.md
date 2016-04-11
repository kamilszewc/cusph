
License
=======

Copyright (c) 2014-2016 Kamil Szewc et al. (Institute of Fluid-Flow Machinery, PAS)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
  * Redistribution of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistribution in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
  * Neither the name of the Institute of Fluid-Flow Machinery, Polish
    Academy of Sciences nor the names of its contributors may be used to endorse
    or promote products derived from this software without specific prior
    written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF METCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUISNESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



Requirements
============

Operating systems
-----------------

The folowing operating systems has been verified:
  * Windows Vista, Windows 7, Windows 8/8.1, Windows 10
  * Linux Ubuntu 12.04, Debian 7, OpenSUSE 13.2, CentOS 7

Graphic cards
-------------
All GPUs supporting CUDA technology should work properly.
CuSPH2d is designed to properly work only with the newest NVIDIA graphic cards.
Therefore, it should be possible to compile cuSPH2d with -arch >= sm_30 option (can be changed in Makefile).
Full list of CUDA-enabled GPU cards can be find under the link:
https://developer.nvidia.com/cuda-gpus 

CUDA Toolkit
------------
To compile the cuSPH2d source code you need install the CUDA Toolkit from
the CUDA Zone:
https://developer.nvidia.com/category/zone/cuda-zone
However, the cuSPH2d sources has been complied and tested only with
the CUDA Toolkits 3.0, 5.0, 5.5, 6.0, 6.5 and 7.0.

Thrust Library
--------------
Some parts of the code use the Thrust library (sorting).
The Thrust library (v.1.4.0) has been fully featured in CUDA Toolkit since
version 4.0. However, if there is no possibility to use one of newer CUDA
Toolkits, the Thrust library can be found in http://thrust.github.io/.
Its confguration consists in unzipping the source code into the folder
'thrust' created in the main cuSPH2d directory.


List of verified configurations
===============================

The cuSPH2d has been tested with the following machines:
 * GeForce GTX 650 1GB, Windows 8, CUDA Toolkit 5.5/6.0 (-arch=sm_30).
 * GeForce GTX 970, Windows 10, CUDA Toolkit 7.0 (-arch=sm_52)
 * Geforce Titan Black, Windows 10, Linux OpenSUSE 13.2, CUDA Toolkit 7 (-arch=sm_35)


Authors
=======

 * Kamil Szewc, PhD 
 * Michal Olejnik, MSc



