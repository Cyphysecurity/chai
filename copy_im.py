import os
names = ["n008-2018-07-27-12-07-38-0400__CAM_BACK_RIGHT__1532708234428113.jpg",
"n008-2018-08-29-16-04-13-0400__CAM_BACK_RIGHT__1535573292928113.jpg",
"n008-2018-08-29-16-04-13-0400__CAM_BACK_RIGHT__1535573571878113.jpg",
"n008-2018-08-31-11-56-46-0400__CAM_BACK_RIGHT__1535731295178113.jpg",
"n008-2018-08-29-16-04-13-0400__CAM_BACK_RIGHT__1535573571878113.jpg",
"n008-2018-08-29-16-04-13-0400__CAM_BACK_RIGHT__1535573085378113.jpg",
"n008-2018-07-27-12-07-38-0400__CAM_BACK_RIGHT__1532707812928113.jpg"]

origin = "/home/cyphysecurity/Documents/llm/DriveLM/DriveLM/challenge/llama_adapter_v2_multimodal7b/data/nuscenes/samples/CAM_BACK_RIGHT/"
dest = "/home/cyphysecurity/Documents/llm/DriveLM/DriveLM/challenge/llama_adapter_v2_multimodal7b/DriveLM_dataset/nuscenes/samples/CAM_BACK_RIGHT/"
for name in names:
    os.system(f"cp {origin}{name} {dest}{name}")