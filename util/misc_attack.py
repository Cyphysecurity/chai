import json
import cv2
import copy

class DatabaseLoadCustom():
    def __init__(self, db_path, llm_name, use_scenetap=False):
        with open(db_path, 'r') as f:
            data_all = json.load(f)
        self.data = data_all
        self.use_scenetap = use_scenetap
        self.llm_name = llm_name
    
    def get_item(self, idx):
        filename = copy.deepcopy(self.data[idx]['image'])
        

        if self.use_scenetap:
            fl_sp = filename[0].split('/')
            fl = ""
            for n in range(len(fl_sp)-1):
                fl += fl_sp[n]
                fl += "/"
            if self.llm_name == "gpt":
                filename[0] = fl + "gpt/" + fl_sp[-1]
            else:
                filename[0] = fl + "internvl/" + fl_sp[-1]
        image_all = []
        for img_path in filename:
            image_raw = cv2.imread(img_path)
            image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            image_all.append(image_raw)
            
        return image_all, self.data[idx]
    

    
    def get_size(self):
        return len(self.data)
