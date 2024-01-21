# Comparative Analysis of Extended Kalman Filter and Unscented Kalman Filter for Simplified Vehicle Localization System

## Author
Hari Prasanth S.M (hpsvm@kth.se)

### Info
This is my final project submission for the course EL2320 Applied Estimation. In this project, the available kitti dataset is used.

Using the GPS and IMU sensor data, two estimations methods are implemented to solve the localization problem and their results are compared.

All the source codes, and algorithm implemetation can be found in the src folder. They are all '.py' files.

For analysing each data file, jupyter notebook is used. The analysis and their results can be found in src folder as '.ipynb' files. Each dataset has their own notebooks.

### How to use?
- To visualize and duplicate my results, just open the notbook files (no need to run)

### Download KITTI RawData

At any case, if there is any problem with the dataset, it can be downloaded again using the following guidelines.

Donwload the following dataset from the link [Kitti data](https://www.cvlibs.net/datasets/kitti/raw_data.php?type=residential)

Download the `synced+rectified data` and `calibration` files for the following 6 dataset:
	- 2011_09_26_0019
	- 2011_09_26_0023
	- 2011_09_26_0035
	- 2011_09_30_0020
	- 2011_09_30_0033
	- 2011_09_30_0034
	
Extract the data into the data folder. You need to have the following folder structure to run the python files:
Project directory:
- data
	-kitti
		- 2011_09_26
			- 2011_09_26_drive_0019_sync
				- <Extracted files>
			- 2011_09_26_drive_0023_sync
				- <Extracted files>
			- 2011_09_26_drive_0035_sync
				- <Extracted files>
			- calib_imu_to_velo.txt
		- 2011_09_30
			- 2011_09_30_drive_0020_sync
				- <Extracted files>
			- 2011_09_30_drive_0033_sync
				- <Extracted files>
			- 2011_09_30_drive_0034_sync
				- <Extracted files>
			- calib_imu_to_velo.txt
	- src
		- <Python src files and notebooks>
		
		


