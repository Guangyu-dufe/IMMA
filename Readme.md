# File Description

## Data_alignment_SD
Aligns Event and Sensor data

## TrafficStream

### main.py
Modified version of TrafficStream, used for LargeST-Stream series data
- **Data splitting**: Preserves entire year of data rather than the original approach of keeping only one month
- **Data processing**: Added nan-to-num function

### main_detect.py
Changes TrafficStream's Detect strategy to Event-based, selecting nodes where Events occur most frequently as ER nodes. See `TrafficStream/src/model/detect.py` line 58-79 for more details

### main_mixure.py
Processes `[T,N]` into `[B,T,N]` and inputs into two different models based on different event_type_code. I am currently modifying this file

# Data
Download the processed SD data from this [[link](https://drive.google.com/drive/folders/149LGHIf_kigVJIFOTsZytGBLxIXPyGUE?usp=sharing)], which contains `['x', 'event_type_code', 'event_severity', 'event_description']`. All data has the shape `[T,N]`

Run the following code to extract the data to `TrafficStream/data/` or `TrafficStream_Detect/data/`

```bash
tar -xzvf SD.tar.gz
```
