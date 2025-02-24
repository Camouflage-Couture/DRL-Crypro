

Candlesticks image data download: https://drive.google.com/file/d/1qX8K8CRnS3lmhA0Trd1Yk-i7_Cdchwbx/view?usp=sharing  
Copy the unzipped folder "candlesticks" to folder "images"  

## Directories of folders are

-- drl-candlesticks-trader  
&emsp;|  
&emsp;-- code  
&emsp;&emsp;|  
&emsp;&emsp;-- libs  
&emsp;|  
&emsp;-- data  
&emsp;|  
&emsp;-- images  
&emsp;&emsp;|   
&emsp;&emsp;-- candlesticks  
&emsp;|  
&emsp;-- results  
&emsp;&emsp;|  
&emsp;&emsp;-- test  
&emsp;&emsp;&emsp;|  
&emsp;&emsp;&emsp;-- 2022-06-19  
&emsp;&emsp;&emsp;|  
&emsp;&emsp;&emsp;-- 2022-08-15  
&emsp;&emsp;|  
&emsp;&emsp;-- train  
&emsp;&emsp;|  
&emsp;&emsp;-- valid  
&emsp;|  
&emsp;-- runs  
&emsp;|  
&emsp;-- weights  

## Run Example

Train all PPO agents in GPU with multi-resolution raw numerical data  
Type following command in the terminal: 
$ python code/main.py -t train -g ppo -r multi  
