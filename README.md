# CHAI

This repository implements the code for the paper:
> **Structured Command Hijacking against Embodied Artificial Intelligence with Text-based Controls**
>
> Luis Burbano, Diego Ortiz, Qi Sun, Siwei Yang, Haoqin Tu, Cihang Xie, Yinzhi Cao, Alvaro A Cardenas
>


## Instructions

This branch implements the attack against [CloudTrack](https://github.com/yblei/CloudTrack). To run this experiment, please see the install CloudTrack by following the instructions.


The command to run the experiments,
```
python main.m --iterations 20 --json assets/training/cars_json.json 
```
By default, the code runs the optimization. For testing you can add the option `--testing`.


## Notes

Explanation for different branches:
|Branch| Explanation|
|--|--|
|[main](https://github.com/Cyphysecurity/chai)|Implements CHAI attacks against CloudTrack|
|[drivelm](https://github.com/Cyphysecurity/chai/tree/drivelm)|Implements CHAI attacks against DriveLM|
|[landing](https://github.com/Cyphysecurity/chai/tree/landing)|Implements CHAI attacks against the emergency drone landing|

We have deleted the story of the repository to avoid leaking our API key.